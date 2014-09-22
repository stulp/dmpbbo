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

#include <boost/serialization/export.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include "functionapproximators/FunctionApproximatorRBFN.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::FunctionApproximatorRBFN);

#include "functionapproximators/ModelParametersRBFN.hpp"
#include "functionapproximators/MetaParametersRBFN.hpp"
#include "functionapproximators/BasisFunction.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"
#include "dmpbbo_io/EigenBoostSerialization.hpp"

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
}

FunctionApproximatorRBFN::FunctionApproximatorRBFN(const ModelParametersRBFN *const model_parameters) 
:
  FunctionApproximator(model_parameters)
{
}


FunctionApproximator* FunctionApproximatorRBFN::clone(void) const {
  // All error checking and cloning is left to the FunctionApproximator constructor.
  return new FunctionApproximatorRBFN(
    dynamic_cast<const MetaParametersRBFN*>(getMetaParameters()),
    dynamic_cast<const ModelParametersRBFN*>(getModelParameters())
    );
};

void FunctionApproximatorRBFN::train(const MatrixXd& inputs, const MatrixXd& targets)
{
  if (isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorRBFN::train more than once. Doing nothing." << endl;
    cerr << "   (if you really want to retrain, call reTrain function instead)" << endl;
    return;
  }
  
  assert(inputs.rows() == targets.rows());
  assert(inputs.cols()==getExpectedInputDim());

  const MetaParametersRBFN* meta_parameters_lwr = 
    dynamic_cast<const MetaParametersRBFN*>(getMetaParameters());
  
  VectorXd min = inputs.colwise().minCoeff();
  VectorXd max = inputs.colwise().maxCoeff();
  
  MatrixXd centers, widths, activations;
  meta_parameters_lwr->getCentersAndWidths(min,max,centers,widths);

  bool normalized_basis_functions=false;  
  bool asymmetric_kernels=false;  
  BasisFunction::Gaussian::activations(centers,widths,inputs,activations,
    normalized_basis_functions,asymmetric_kernels);
  
  // The design matrix
  MatrixXd X = activations;

  // Least squares
  VectorXd weights = (X.transpose()*X).inverse()*X.transpose()*targets;

  setModelParameters(new ModelParametersRBFN(centers,widths,weights));
  
}

void FunctionApproximatorRBFN::predict(const MatrixXd& inputs, MatrixXd& output)
{
  if (!isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorLWPR::predict if you have not trained yet. Doing nothing." << endl;
    return;
  }

  const ModelParametersRBFN* model_parameters_rbfn = static_cast<const ModelParametersRBFN*>(getModelParameters());
  
  output.resize(inputs.rows(),1); // Fix this
  // Assert that memory has been pre-allocated.
  assert(inputs.rows()==output.rows());
  
  // Get the basis function activations  
  MatrixXd activations; // todo avoid allocation
  model_parameters_rbfn->kernelActivations(inputs,activations);
    
  // Weight the basis function activations  
  VectorXd weights = model_parameters_rbfn->weights();
  for (int b=0; b<activations.cols(); b++)
    activations.col(b).array() *= weights(b);

  // Sum over weighed basis functions
  output = activations.rowwise().sum();
    
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

template<class Archive>
void FunctionApproximatorRBFN::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(FunctionApproximator);
}

}
