/**
 * @file   FunctionApproximatorGPR.cpp
 * @brief  FunctionApproximatorGPR class source file.
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
#include "functionapproximators/FunctionApproximatorGPR.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::FunctionApproximatorGPR);

#include "functionapproximators/ModelParametersGPR.hpp"
#include "functionapproximators/MetaParametersGPR.hpp"
#include "functionapproximators/BasisFunction.hpp"

#include "dmpbbo_io/EigenBoostSerialization.hpp"
#include "dmpbbo_io/EigenFileIO.hpp"

#include <iostream>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/LU>

using namespace std;
using namespace Eigen;

namespace DmpBbo {

FunctionApproximatorGPR::FunctionApproximatorGPR(const MetaParametersGPR *const meta_parameters, const ModelParametersGPR *const model_parameters) 
:
  FunctionApproximator(meta_parameters,model_parameters)
{
}

FunctionApproximatorGPR::FunctionApproximatorGPR(const ModelParametersGPR *const model_parameters) 
:
  FunctionApproximator(model_parameters)
{
}


FunctionApproximator* FunctionApproximatorGPR::clone(void) const {
  // All error checking and cloning is left to the FunctionApproximator constructor.
  return new FunctionApproximatorGPR(
    dynamic_cast<const MetaParametersGPR*>(getMetaParameters()),
    dynamic_cast<const ModelParametersGPR*>(getModelParameters())
    );
};

void FunctionApproximatorGPR::train(const MatrixXd& inputs, const MatrixXd& targets)
{
  if (isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorGPR::train more than once. Doing nothing." << endl;
    cerr << "   (if you really want to retrain, call reTrain function instead)" << endl;
    return;
  }
  
  assert(inputs.rows() == targets.rows());
  assert(inputs.cols()==getExpectedInputDim());

  const MetaParametersGPR* meta_parameters_gpr = 
    dynamic_cast<const MetaParametersGPR*>(getMetaParameters());
  
  double max_covar = meta_parameters_gpr->maximum_covariance();
  VectorXd sigmas = meta_parameters_gpr->sigmas();
  
  
  // Compute the gram matrix
  // In a gram matrix, every input point is itself a center
  MatrixXd centers = inputs;
  // Replicate sigmas, because they are the same for each data point/center
  MatrixXd widths = sigmas.transpose().colwise().replicate(centers.rows()); 

  MatrixXd gram(inputs.rows(),inputs.rows());
  bool normalize_activations = false;
  bool asymmetric_kernels = false;
  BasisFunction::Gaussian::activations(centers,widths,inputs,gram,normalize_activations,asymmetric_kernels);
  
  gram *= max_covar;

  setModelParameters(new ModelParametersGPR(inputs,targets,gram,max_covar,sigmas));
  
}

void FunctionApproximatorGPR::predict(const MatrixXd& inputs, MatrixXd& outputs)
{
  if (!isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorLWPR::predict if you have not trained yet. Doing nothing." << endl;
    return;
  }

  const ModelParametersGPR* model_parameters_gpr = static_cast<const ModelParametersGPR*>(getModelParameters());
  
  assert(inputs.cols()==getExpectedInputDim());
  unsigned int n_samples = inputs.rows();
  
  outputs.resize(n_samples,1);
  
  MatrixXd ks;
  model_parameters_gpr->kernelActivations(inputs, ks);
  
  VectorXd weights = model_parameters_gpr->weights();
  for (unsigned int ii=0; ii<n_samples; ii++)
    outputs(ii) = ks.row(ii).dot(weights);
  
}

void FunctionApproximatorGPR::predictVariance(const MatrixXd& inputs, MatrixXd& variances)
{
  if (!isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorLWPR::predict if you have not trained yet. Doing nothing." << endl;
    return;
  }

  const ModelParametersGPR* model_parameters_gpr = static_cast<const ModelParametersGPR*>(getModelParameters());
  
  
  assert(inputs.cols()==getExpectedInputDim());
  
  unsigned int n_samples = inputs.rows();
  variances.resize(n_samples,1);
  
  MatrixXd ks;
  model_parameters_gpr->kernelActivations(inputs, ks);  

  double maximum_covariance = model_parameters_gpr->maximum_covariance();
  MatrixXd gram_inv = model_parameters_gpr->gram_inv();
  
  for (unsigned int ii=0; ii<n_samples; ii++)
    variances(ii) = maximum_covariance - (ks.row(ii)*gram_inv).dot(ks.row(ii).transpose());

}

bool FunctionApproximatorGPR::saveGridData(const VectorXd& min, const VectorXd& max, const VectorXi& n_samples_per_dim, string save_directory, bool overwrite) const
{
  if (save_directory.empty())
    return true;
  
  MatrixXd inputs_grid;
  FunctionApproximator::generateInputsGrid(min, max, n_samples_per_dim, inputs_grid);
      
  const ModelParametersGPR* model_parameters_gpr = static_cast<const ModelParametersGPR*>(getModelParameters());

  MatrixXd activations_grid;
  model_parameters_gpr->kernelActivations(inputs_grid, activations_grid);
  
  saveMatrix(save_directory,"n_samples_per_dim.txt",n_samples_per_dim,overwrite);
  saveMatrix(save_directory,"inputs_grid.txt",inputs_grid,overwrite);
  saveMatrix(save_directory,"activations_grid.txt",activations_grid,overwrite);

  // Weight the basis function activations  
  VectorXd weights = model_parameters_gpr->weights();
  for (int b=0; b<activations_grid.cols(); b++)
    activations_grid.col(b).array() *= weights(b);
  saveMatrix(save_directory,"activations_weighted_grid.txt",activations_grid,overwrite);
  
  // Sum over weighed basis functions
  MatrixXd predictions_grid = activations_grid.rowwise().sum();
  saveMatrix(save_directory,"predictions_grid.txt",predictions_grid,overwrite);
  
  return true;
  
}

template<class Archive>
void FunctionApproximatorGPR::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(FunctionApproximator);
}

}
