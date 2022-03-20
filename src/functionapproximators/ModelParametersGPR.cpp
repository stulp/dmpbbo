/**
 * @file   ModelParametersGPR.cpp
 * @brief  ModelParametersGPR class source file.
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
 
#include "functionapproximators/ModelParametersGPR.hpp"
#include "functionapproximators/UnifiedModel.hpp"

#include "functionapproximators/BasisFunction.hpp"
#include "dmpbbo_io/EigenFileIO.hpp"
#include "dmpbbo_io/BoostSerializationToString.hpp"

#include <iostream>
#include <fstream>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/LU>


using namespace std;
using namespace Eigen;

namespace DmpBbo {

ModelParametersGPR::ModelParametersGPR(const Eigen::MatrixXd& train_inputs, const Eigen::VectorXd& train_targets, const Eigen::MatrixXd& gram, double maximum_covariance, double length)
:
  train_inputs_(train_inputs),
  train_targets_(train_targets),
  gram_(gram),
  maximum_covariance_(maximum_covariance),
  sigmas_(VectorXd::Constant(train_inputs_.cols(),length))
{
  assert(gram_.rows()==gram_.cols());
  assert(train_inputs_.rows()==gram_.rows());
  assert(train_inputs_.rows()==train_targets_.rows());
  assert(maximum_covariance_>0);
  assert(length>0);
  
  gram_inv_ = gram_.inverse();
  gram_inv_targets_ = gram_inv_ * train_targets_.col(0);

}

ModelParametersGPR::ModelParametersGPR(const Eigen::MatrixXd& train_inputs, const Eigen::VectorXd& train_targets, const Eigen::MatrixXd& gram, double maximum_covariance, const Eigen::VectorXd& sigmas)
:
  train_inputs_(train_inputs),
  train_targets_(train_targets),
  gram_(gram),
  maximum_covariance_(maximum_covariance),
  sigmas_(sigmas)
{
  assert(gram_.rows()==gram_.cols());
  assert(train_inputs_.rows()==gram_.rows());
  assert(train_inputs_.rows()==train_targets_.rows());
  assert(sigmas_.size()==train_inputs_.cols());
  assert(maximum_covariance_>0);
  
  gram_inv_ = gram_.inverse();
  gram_inv_targets_ = gram_inv_ * train_targets_.col(0);

}

ModelParameters* ModelParametersGPR::clone(void) const {
  return new ModelParametersGPR(train_inputs_,train_targets_,gram_,maximum_covariance_,sigmas_); 
}

void ModelParametersGPR::kernelActivations(const Eigen::Ref<const Eigen::MatrixXd>& inputs, Eigen::MatrixXd& kernel_activations) const
{
  
  MatrixXd centers = train_inputs_;
  int n_basis_functions = centers.rows();
  // All basis functions have the same width
  MatrixXd widths  = sigmas_.transpose().colwise().replicate(n_basis_functions); 
  
  bool normalize_activations = false;
  bool asymmetric_kernels = false;
  BasisFunction::Gaussian::activations(centers,widths,inputs,kernel_activations,normalize_activations,asymmetric_kernels);
  
  kernel_activations *= maximum_covariance_;
}

string ModelParametersGPR::toString(void) const
{
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("ModelParametersGPR");
}

void ModelParametersGPR::getSelectableParameters(set<string>& selected_values_labels) const 
{
  selected_values_labels = set<string>();
}

UnifiedModel* ModelParametersGPR::toUnifiedModel(void) const
{
 
  MatrixXd centers = train_inputs_;
  int n_basis_functions = centers.rows();
  // All basis functions have the same width
  MatrixXd widths  = sigmas_.transpose().colwise().replicate(n_basis_functions); 
  MatrixXd weights = gram_inv_targets_*maximum_covariance_;
  bool normalized_basis_functions = false;

  return new UnifiedModel(centers, widths, weights, normalized_basis_functions); 
  
}


}
