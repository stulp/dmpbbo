/**
 * @file   ModelParametersRBFN.cpp
 * @brief  ModelParametersRBFN class source file.
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

#include "functionapproximators/ModelParametersRBFN.hpp"
#include "functionapproximators/BasisFunction.hpp"
#include "functionapproximators/UnifiedModel.hpp"

#include "dmpbbo_io/BoostSerializationToString.hpp"

#include <iostream>
#include <fstream>

#include <eigen3/Eigen/Core>


using namespace std;
using namespace Eigen;

namespace DmpBbo {

ModelParametersRBFN::ModelParametersRBFN(const Eigen::MatrixXd& centers, const Eigen::MatrixXd& widths, const Eigen::MatrixXd& weights) 
:
  centers_(centers),
  widths_(widths),
  weights_(weights),
  caching_(false)
{
  int n_basis = centers.rows();
  int n_dims = centers.cols();

  assert(n_basis==widths_.rows());
  assert(n_dims ==widths_.cols());
  assert(n_basis==weights_.rows());
  assert(1      ==weights_.cols());
  
  min_["centers"] = centers_.minCoeff();
  max_["centers"] = centers_.maxCoeff();
  min_["widths"] = widths_.minCoeff();
  max_["widths"] = widths_.maxCoeff();
  min_["weights"] = weights_.minCoeff();
  max_["weights"] = weights_.maxCoeff();
  checkMinMax();

  sizes_["centers"] = n_dims*n_basis;
  sizes_["widths"] = n_dims*n_basis;
  sizes_["weights"] = 1*n_basis;

};

ModelParameters* ModelParametersRBFN::clone(void) const {
  return new ModelParametersRBFN(centers_,widths_,weights_); 
}

void ModelParametersRBFN::kernelActivations(const Eigen::Ref<const Eigen::MatrixXd>& inputs, Eigen::MatrixXd& kernel_activations) const
{
  if (caching_)
  {
    // If the cached inputs matrix has the same size as the one now requested
    //     (this also takes care of the case when inputs_cached is empty and need to be initialized)
    if ( inputs.rows()==inputs_cached_.rows() && inputs.cols()==inputs_cached_.cols() )
    {
      // And they have the same values
      if ( (inputs.array()==inputs_cached_.array()).all() )
      {
        // Then you can return the cached values
        kernel_activations = kernel_activations_cached_;
        return;
      }
    }
  }
  
  ENTERING_REAL_TIME_CRITICAL_CODE

  // Cache could not be used, actually do the work
  bool normalized_basis_functions=false;  
  bool asymmetric_kernels=false;
  BasisFunction::Gaussian::activations(centers_,widths_,inputs,kernel_activations,
    normalized_basis_functions,asymmetric_kernels);
  
  EXITING_REAL_TIME_CRITICAL_CODE

  if (caching_)
  {
    // Cache the current results now.  
    inputs_cached_ = inputs;
    kernel_activations_cached_ = kernel_activations;
  }
  
}

string ModelParametersRBFN::toString(void) const
{
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("ModelParametersRBFN");
}


void ModelParametersRBFN::getSelectableParameters(set<string>& labels) const 
{
  labels = set<string>();
  labels.insert("centers");
  labels.insert("widths");
  labels.insert("weights");
}

void ModelParametersRBFN::getParameterVector(Eigen::VectorXd& values, bool normalized) const
{
  
  values.resize(getParameterVectorSize());
  
  int n_dims = getExpectedInputDim();
  unsigned int n_basis = getNumberOfBasisFunctions();
  int offset = 0;
  
  double min = 0.0;
  double max = 1.0;
  
  string label = "centers";
  if (isParameterSelected(label)) {
    if (normalized) {
      min = min_.at(label);
      max = max_.at(label);
    }
    
    for (int i_dim=0; i_dim<n_dims; i_dim++)
      values.segment(offset+i_dim*n_basis,n_basis) =
        (centers_.col(i_dim).array()-min)/(max-min);
        
    offset += n_dims*n_basis;
  }
  
  label = "widths";
  if (isParameterSelected(label)) {
    if (normalized) {
      min = min_.at(label);
      max = max_.at(label);
    }
    
    for (int i_dim=0; i_dim<n_dims; i_dim++)
      values.segment(offset+i_dim*n_basis,n_basis) = 
        (widths_.col(i_dim).array()-min)/(max-min);
        
    offset += n_dims*n_basis;
  }
  
  label = "weights";
  if (isParameterSelected(label)) {
    if (normalized) {
      min = min_.at(label);
      max = max_.at(label);
    }
    values.segment(offset,n_basis) = 
        (weights_.array()-min)/(max-min);
    offset += n_basis;
  }
  
  assert(offset == getParameterVectorSize());   
};

void ModelParametersRBFN::setParameterVector(const VectorXd& values, bool normalized) {

  int expected_size = getParameterVectorSize();
  if (expected_size != values.size())
  {
    cerr << __FILE__ << ":" << __LINE__ << ": values is of wrong size." << endl;
    return;
  }
  
  int n_dims = getExpectedInputDim();
  unsigned int n_basis = getNumberOfBasisFunctions();
  int offset = 0;

  string l = "centers";
  if (isParameterSelected(l)) {
    for (int i_dim=0; i_dim<n_dims; i_dim++)
      centers_.col(i_dim) = values.segment(offset+i_dim*n_basis,n_basis);
    if (normalized)
      centers_ = ((max_[l]-min_[l])*centers_.array())+min_[l];
    offset += n_dims*n_basis;
    clearCache(); // Centers updated, activation need to be updated.
  }
  
  l = "widths";
  if (isParameterSelected(l)) {
    for (int i_dim=0; i_dim<n_dims; i_dim++)
      widths_.col(i_dim) = values.segment(offset+i_dim*n_basis,n_basis);
    if (normalized)
      widths_ = ((max_[l]-min_[l])*widths_.array())+min_[l];
    offset += n_dims*n_basis;
    clearCache(); // Centers updated, activation need to be updated.
  }
  
  l = "weights";
  if (isParameterSelected(l)) {
    weights_ = values.segment(offset,n_basis);
    if (normalized)
      weights_ = ((max_[l]-min_[l])*weights_.array())+min_[l];
    offset += n_basis;
    // Cache must not be cleared, because kernelActivations() returns the same values.
  }

  assert(offset == expected_size);   
};

void ModelParametersRBFN::setParameterVectorModifierPrivate(std::string modifier, bool new_value)
{
}

UnifiedModel* ModelParametersRBFN::toUnifiedModel(void) const
{
  // RBFN does not use normalized basis functions
  bool normalized_basis_functions = false;
  return new UnifiedModel(centers_, widths_, weights_,normalized_basis_functions); 
}

}


