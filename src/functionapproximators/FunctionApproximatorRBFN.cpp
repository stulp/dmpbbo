/**
 * @file   FunctionApproximatorRBFN.cpp
 * @brief  FunctionApproximatorRBFN class source file.
 * @author Freek Stulp
 *
 * This file is part of DmpBbo, a set of libraries and programs for the 
 * black-box optimization of dynamical movement primitives.
 * Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
 * 
 * DmpBbo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 * 
 * DmpBbo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "functionapproximators/FunctionApproximatorRBFN.hpp"
#include "functionapproximators/ModelParametersRBFN.hpp"
#include "functionapproximators/MetaParametersRBFN.hpp"
#include "functionapproximators/BasisFunction.hpp"
#include "functionapproximators/leastSquares.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"
#include "dmpbbo_io/BoostSerializationToString.hpp"

#include <iostream>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/LU>

using namespace std;
using namespace Eigen;

namespace DmpBbo {

FunctionApproximatorRBFN::FunctionApproximatorRBFN(const MetaParametersRBFN *const meta_parameters, const ModelParametersRBFN *const model_parameters) 
:
  FunctionApproximator(meta_parameters,model_parameters)
{
  if (model_parameters!=NULL)
    preallocateMemory(model_parameters->getNumberOfBasisFunctions());
}

FunctionApproximatorRBFN::FunctionApproximatorRBFN(const ModelParametersRBFN *const model_parameters) 
:
  FunctionApproximator(model_parameters)
{
  preallocateMemory(model_parameters->getNumberOfBasisFunctions());
}

void FunctionApproximatorRBFN::preallocateMemory(int n_basis_functions)
{
  weights_prealloc_ = VectorXd(n_basis_functions);
  activations_one_prealloc_ = MatrixXd(1,n_basis_functions);
  activations_prealloc_ = MatrixXd(1,n_basis_functions);
}


FunctionApproximator* FunctionApproximatorRBFN::clone(void) const {
  // All error checking and cloning is left to the FunctionApproximator constructor.
  return new FunctionApproximatorRBFN(
    dynamic_cast<const MetaParametersRBFN*>(getMetaParameters()),
    dynamic_cast<const ModelParametersRBFN*>(getModelParameters())
    );
};

void FunctionApproximatorRBFN::train(const Eigen::Ref<const Eigen::MatrixXd>& inputs, const Eigen::Ref<const Eigen::MatrixXd>& targets)
{
  if (isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorRBFN::train more than once. Doing nothing." << endl;
    cerr << "   (if you really want to retrain, call reTrain function instead)" << endl;
    return;
  }
  
  assert(inputs.rows() == targets.rows());
  assert(inputs.cols()==getExpectedInputDim());

  const MetaParametersRBFN* meta_parameters_rbfn = 
    dynamic_cast<const MetaParametersRBFN*>(getMetaParameters());
                      
  // Determine the centers and widths of the basis functions, given the range of the input data
  VectorXd min = inputs.colwise().minCoeff();
  VectorXd max = inputs.colwise().maxCoeff();
  MatrixXd centers, widths;
  meta_parameters_rbfn->getCentersAndWidths(min,max,centers,widths);

  // Get the activations of the basis functions 
  bool normalized_basis_functions=false;  
  bool asymmetric_kernels=false;  
  int n_samples = inputs.rows();
  int n_kernels = centers.rows(); 
  MatrixXd activations(n_samples,n_kernels);
  BasisFunction::Gaussian::activations(centers,widths,inputs,activations,
    normalized_basis_functions,asymmetric_kernels);

  // Least squares, with activations as design matrix
  bool use_offset=false;
  double regularization = meta_parameters_rbfn->regularization();
  VectorXd weights = leastSquares(activations,targets,use_offset,regularization);

  setModelParameters(new ModelParametersRBFN(centers,widths,weights));
  
  preallocateMemory(n_kernels);

}

void FunctionApproximatorRBFN::predict(const Eigen::Ref<const Eigen::MatrixXd>& inputs, MatrixXd& outputs)
{
  if (!isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorLWPR::predict if you have not trained yet. Doing nothing." << endl;
    return;
  }

  const ModelParametersRBFN* model_parameters_rbfn = static_cast<const ModelParametersRBFN*>(getModelParameters());
  
  model_parameters_rbfn->weights(weights_prealloc_);
  
  int n_basis_functions = model_parameters_rbfn->getNumberOfBasisFunctions();
  
  bool only_one_sample = (inputs.rows()==1);
  if (only_one_sample)
  {
    ENTERING_REAL_TIME_CRITICAL_CODE
    
    // Get the basis function activations  
    model_parameters_rbfn->kernelActivations(inputs,activations_one_prealloc_);
      
    // Weight the basis function activations  
    for (int b=0; b<n_basis_functions; b++)
      activations_one_prealloc_.col(b).array() *= weights_prealloc_(b);
  
    // Sum over weighed basis functions
    outputs = activations_one_prealloc_.rowwise().sum();
    
    EXITING_REAL_TIME_CRITICAL_CODE
  }
  else 
  {
    int n_time_steps = inputs.rows();

    // The next two lines may not be real-time, as they may allocate memory.
    // (if the size are already correct, it will be realtime)
    activations_prealloc_.resize(n_time_steps,n_basis_functions);
    outputs.resize(n_time_steps,getExpectedOutputDim());
    
    // Get the basis function activations  
    model_parameters_rbfn->kernelActivations(inputs,activations_prealloc_);
      
    // Weight the basis function activations  
    for (int b=0; b<n_basis_functions; b++)
      activations_prealloc_.col(b).array() *= weights_prealloc_(b);
  
    // Sum over weighed basis functions
    outputs = activations_prealloc_.rowwise().sum();
  }
    
}

bool FunctionApproximatorRBFN::saveGridData(const VectorXd& min, const VectorXd& max, const VectorXi& n_samples_per_dim, string save_directory, bool overwrite) const
{
  if (save_directory.empty())
    return true;
  
  MatrixXd inputs_grid;
  FunctionApproximator::generateInputsGrid(min, max, n_samples_per_dim, inputs_grid);
      
  const ModelParametersRBFN* model_parameters_rbfn = static_cast<const ModelParametersRBFN*>(getModelParameters());
  
  MatrixXd activations_grid;
  model_parameters_rbfn->kernelActivations(inputs_grid, activations_grid);
  
  saveMatrix(save_directory,"n_samples_per_dim.txt",n_samples_per_dim,overwrite);
  saveMatrix(save_directory,"inputs_grid.txt",inputs_grid,overwrite);
  saveMatrix(save_directory,"activations_grid.txt",activations_grid,overwrite);

  // Weight the basis function activations  
  VectorXd weights = model_parameters_rbfn->weights();
  for (int b=0; b<activations_grid.cols(); b++)
    activations_grid.col(b).array() *= weights(b);
  saveMatrix(save_directory,"activations_weighted_grid.txt",activations_grid,overwrite);
  
  // Sum over weighed basis functions
  MatrixXd predictions_grid = activations_grid.rowwise().sum();
  saveMatrix(save_directory,"predictions_grid.txt",predictions_grid,overwrite);
  
  return true;
  
}


}
