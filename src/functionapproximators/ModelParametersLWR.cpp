/**
 * @file   ModelParametersLWR.cpp
 * @brief  ModelParametersLWR class source file.
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
 
#include "functionapproximators/ModelParametersLWR.hpp"
#include "functionapproximators/UnifiedModel.hpp"
#include "functionapproximators/BasisFunction.hpp"

#include "dmpbbo_io/BoostSerializationToString.hpp"

#include <iostream>
#include <fstream>

#include <eigen3/Eigen/Core>


using namespace std;
using namespace Eigen;

namespace DmpBbo {

ModelParametersLWR::ModelParametersLWR(const Eigen::MatrixXd& centers, const Eigen::MatrixXd& widths, const Eigen::MatrixXd& slopes, const Eigen::MatrixXd& offsets, bool asymmetric_kernels, bool lines_pivot_at_max_activation) 
:
  centers_(centers),
  widths_(widths),
  slopes_(slopes), 
  offsets_(offsets),
  asymmetric_kernels_(asymmetric_kernels),
  lines_pivot_at_max_activation_(lines_pivot_at_max_activation),
  slopes_as_angles_(false),
  caching_(false)
{
  int n_basis = centers.rows();
  int n_dims = centers.cols();

  assert(n_basis==widths_.rows());
  assert(n_dims ==widths_.cols());
  assert(n_basis==slopes_.rows());
  assert(n_dims ==slopes_.cols());
  assert(n_basis==offsets_.rows());
  assert(1      ==offsets_.cols());
  
  min_["centers"] = centers_.minCoeff();
  max_["centers"] = centers_.maxCoeff();
  min_["widths"] = widths_.minCoeff();
  max_["widths"] = widths_.maxCoeff();
  min_["slopes"] = slopes_.minCoeff();
  max_["slopes"] = slopes_.maxCoeff();
  min_["offsets"] = offsets_.minCoeff();
  max_["offsets"] = offsets_.maxCoeff();
  checkMinMax();
  
  /*
  cout << "==========" << endl;
  cout << "centers_ = " << centers_ << endl;
  cout << "   min= " << min_["centers"] << " " << centers_.minCoeff() << endl;
  cout << "   max = " << max_["centers"] << " " << centers_.maxCoeff() << endl;
  cout << "widths_ = " << widths_ << endl;
  cout << "   min= " << min_["widths"] << " " << widths_.minCoeff() << endl;
  cout << "   max = " << max_["widths"] << " " << widths_.maxCoeff() << endl;
  cout << "slopes_ = " << slopes_ << endl;
  cout << "   min= " << min_["slopes"] << " " << slopes_.minCoeff() << endl;
  cout << "   max = " << max_["slopes"] << " " << slopes_.maxCoeff() << endl;
  cout << "offsets_ = " << offsets_ << endl;
  cout << "   min= " << min_["offsets"] << " " << offsets_.minCoeff() << endl;
  cout << "   max = " << max_["offsets"] << " " << offsets_.maxCoeff() << endl;
  */

  sizes_["centers"] = n_dims*n_basis;
  sizes_["widths"] = n_dims*n_basis;
  sizes_["slopes"] = n_dims*n_basis;
  sizes_["offsets"] = 1*n_basis;

};

ModelParameters* ModelParametersLWR::clone(void) const {
  return new ModelParametersLWR(centers_,widths_,slopes_,offsets_,asymmetric_kernels_,lines_pivot_at_max_activation_); 
}

void ModelParametersLWR::unnormalizedKernelActivations(const Eigen::Ref<const Eigen::MatrixXd>& inputs, Eigen::MatrixXd& kernel_activations) const
{
  ENTERING_REAL_TIME_CRITICAL_CODE
  bool normalized_basis_functions=false;  
  BasisFunction::Gaussian::activations(centers_,widths_,inputs,kernel_activations,
    normalized_basis_functions,asymmetric_kernels_);  
  EXITING_REAL_TIME_CRITICAL_CODE
}

void ModelParametersLWR::kernelActivations(const Eigen::Ref<const Eigen::MatrixXd>& inputs, Eigen::MatrixXd& kernel_activations) const
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
  bool normalize_activations = true;
  BasisFunction::Gaussian::activations(centers_,widths_,inputs,kernel_activations,normalize_activations,asymmetric_kernels_);

  EXITING_REAL_TIME_CRITICAL_CODE
  
  if (caching_)
  {
    // Cache the current results now.  
    inputs_cached_ = inputs;
    kernel_activations_cached_ = kernel_activations;
  }
  
}

void ModelParametersLWR::set_lines_pivot_at_max_activation(bool lines_pivot_at_max_activation)
{
  // If no change, just return
  if (lines_pivot_at_max_activation_ == lines_pivot_at_max_activation)
    return;

  //cout << "________________" << endl;
  //cout << centers_.transpose() << endl;  
  //cout << slopes_.transpose() << endl;  
  //cout << offsets_.transpose() << endl;  
  //cout << "centers_ = " << centers_.rows() << "X" << centers_.cols() << endl;
  //cout << "slopes_ = " << slopes_.rows() << "X" << slopes_.cols() << endl;
  //cout << "offsets_ = " << offsets_.rows() << "X" << offsets_.cols() << endl;

  // If you pivot lines around the point when the basis function has maximum activation (i.e.
  // at the center of the Gaussian), you must compute the new offset corresponding to this
  // slope, and vice versa    
  int n_lines = centers_.rows();
  VectorXd ac(n_lines); // slopes*centers
  for (int i_line=0; i_line<n_lines; i_line++)
  {
    ac[i_line] = slopes_.row(i_line) * centers_.row(i_line).transpose();
  }
    
  if (lines_pivot_at_max_activation)
  {
    // Representation was "y = ax + b", now it will be "y = a(x-c) + b^new" 
    // Since "y = ax + b" can be rewritten as "y = a(x-c) + (b+ac)", we know that "b^new = (ac+b)"
    offsets_ = offsets_ + ac;
  }
  else
  {
    // Representation was "y = a(x-c) + b", now it will be "y = ax + b^new" 
    // Since "y = a(x-c) + b" can be rewritten as "y = ax + (b-ac)", we know that "b^new = (b-ac)"
    offsets_ = offsets_ - ac;
  } 
  // Remark, the above could have been done as a one-liner, but I prefer the more legible version.
  
  //cout << offsets_.transpose() << endl;  
  //cout << "offsets_ = " << offsets_.rows() << "X" << offsets_.cols() << endl;
  
  lines_pivot_at_max_activation_ = lines_pivot_at_max_activation;
}

void ModelParametersLWR::set_slopes_as_angles(bool slopes_as_angles)
{
  slopes_as_angles_ = slopes_as_angles;
  cerr << __FILE__ << ":" << __LINE__ << ":";
  cerr << "Not implemented yet!!!" << endl;
  slopes_as_angles_ = false;
}

/*
The code below was previously implemted as follows. Below is the real-time version.


void ModelParametersLWR::getLines(const Eigen::Ref<const Eigen::MatrixXd>& inputs, MatrixXd& lines) const
{
  int n_time_steps = inputs.rows();

  //cout << "centers_ = " << centers_.rows() << "X" << centers_.cols() << endl;
  //cout << "slopes_ = " << slopes_.rows() << "X" << slopes_.cols() << endl;
  //cout << "offsets_ = " << offsets_.rows() << "X" << offsets_.cols() << endl;
  //cout << "inputs = " << inputs.rows() << "X" << inputs.cols() << endl;
  
  // Compute values along lines for each time step  
  // Line representation is "y = ax + b"
  lines = inputs*slopes_.transpose() + offsets_.transpose().replicate(n_time_steps,1);
  
  if (lines_pivot_at_max_activation_)
  {
    // Line representation is "y = a(x-c) + b", which is  "y = ax - ac + b"
    // Therefore, we still have to subtract "ac"
    int n_lines = centers_.rows();
    VectorXd ac(n_lines); // slopes*centers  = ac
    for (int i_line=0; i_line<n_lines; i_line++)
      ac[i_line] = slopes_.row(i_line) * centers_.row(i_line).transpose();
    //cout << "ac = " << ac.rows() << "X" << ac.cols() << endl;
    lines = lines - ac.transpose().replicate(n_time_steps,1);
  }
  //cout << "lines = " << lines.rows() << "X" << lines.cols() << endl;
}
*/

void ModelParametersLWR::getLines(const Eigen::Ref<const Eigen::MatrixXd>& inputs, MatrixXd& lines) const
{
  ENTERING_REAL_TIME_CRITICAL_CODE
  
  //cout  << endl << "========================" << endl;
  //cout << "lines = " << lines.rows() << "X" << lines.cols() << endl;
  //cout << "centers_ = " << centers_.rows() << "X" << centers_.cols() << endl;
  //cout << "slopes_ = " << slopes_.rows() << "X" << slopes_.cols() << endl;
  //cout << "offsets_ = " << offsets_.rows() << "X" << offsets_.cols() << endl;
  //cout << "inputs = " << inputs.rows() << "X" << inputs.cols() << endl;
  
  int n_time_steps = inputs.rows();
  int n_lines = centers_.rows();
  lines.resize(n_time_steps,n_lines);
  //cout << "lines = " << lines.rows() << "X" << lines.cols() << endl;


  // Compute values along lines for each time step  
  // Line representation is "y = ax + b"
  for (int i_line=0; i_line<n_lines; i_line++)
  {
    lines.col(i_line).noalias() = inputs*slopes_.row(i_line).transpose();
    lines.col(i_line).array() += offsets_(i_line);
  
    if (lines_pivot_at_max_activation_)
    {
      // Line representation is "y = a(x-c) + b", which is  "y = ax - ac + b"
      // Therefore, we still have to subtract "ac"
      double ac = slopes_.row(i_line).dot(centers_.row(i_line));
      lines.col(i_line).array() -= ac;
    }
  }
  
  EXITING_REAL_TIME_CRITICAL_CODE
}

/*
void ModelParametersLWR::kernelActivationsSymmetric(const MatrixXd& centers, const MatrixXd& widths, const Eigen::Ref<const Eigen::MatrixXd>& inputs, MatrixXd& kernel_activations)
{
  cout << __FILE__ << ":" << __LINE__ << ":Here" << endl;
  // Check and set sizes
  // centers     = n_basis_functions x n_dim
  // widths      = n_basis_functions x n_dim
  // inputs      = n_samples         x n_dim
  // activations = n_samples         x n_basis_functions
  int n_basis_functions = centers.rows();
  int n_samples         = inputs.rows();
  int n_dims            = centers.cols();
  assert( (n_basis_functions==widths.rows()) & (n_dims==widths.cols()) ); 
  assert( (n_samples==inputs.rows()        ) & (n_dims==inputs.cols()) ); 
  kernel_activations.resize(n_samples,n_basis_functions);  


  VectorXd center, width;
  for (int bb=0; bb<n_basis_functions; bb++)
  {
    center = centers.row(bb);
    width  = widths.row(bb);

    // Here, we compute the values of a (unnormalized) multi-variate Gaussian:
    //   activation = exp(-0.5*(x-mu)*Sigma^-1*(x-mu))
    // Because Sigma is diagonal in our case, this simplifies to
    //   activation = exp(\sum_d=1^D [-0.5*(x_d-mu_d)^2/Sigma_(d,d)]) 
    //              = \prod_d=1^D exp(-0.5*(x_d-mu_d)^2/Sigma_(d,d)) 
    // This last product is what we compute below incrementally
    
    kernel_activations.col(bb).fill(1.0);
    for (int i_dim=0; i_dim<n_dims; i_dim++)
    {
      kernel_activations.col(bb).array() *= exp(-0.5*pow(inputs.col(i_dim).array()-center[i_dim],2)/(width[i_dim]*width[i_dim])).array();
    }
  }
}
*/


string ModelParametersLWR::toString(void) const
{
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("ModelParametersLWR");
}

void ModelParametersLWR::getSelectableParameters(set<string>& selected_values_labels) const 
{
  selected_values_labels = set<string>();
  selected_values_labels.insert("centers");
  selected_values_labels.insert("widths");
  selected_values_labels.insert("offsets");
  selected_values_labels.insert("slopes");
}


void ModelParametersLWR::getParameterVector(Eigen::VectorXd& values, bool normalized) const
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
  
  
  /*
  VectorXd cur_slopes;
  for (int i_dim=0; i_dim<slopes_.cols(); i_dim++)
  {
    cur_slopes = slopes_.col(i_dim);
    if (slopes_as_angles_)
    {
      // cur_slopes is a slope, but the values vector expects the angle with the x-axis. Do the 
      // conversion here.
      for (int ii=0; ii<cur_slopes.size(); ii++)
        cur_slopes[ii] = atan2(cur_slopes[ii],1.0);
    }
    
    values.segment(offset,slopes_.rows()) = cur_slopes;
    offset += slopes_.rows();
  }
  */
  
  label = "slopes";
  if (isParameterSelected(label)) {
    if (normalized) {
      min = min_.at(label);
      max = max_.at(label);
      cout << label << min << " " << max << endl;
    }
    
    for (int i_dim=0; i_dim<n_dims; i_dim++)
      values.segment(offset+i_dim*n_basis,n_basis) = 
        (slopes_.col(i_dim).array()-min)/(max-min);
        
    offset += n_dims*n_basis;
  }
  
  label = "offsets";
  if (isParameterSelected(label)) {
    if (normalized) {
      min = min_.at(label);
      max = max_.at(label);
      cout << label << min << " " << max << endl;
    }
    values.segment(offset,n_basis) = 
        (offsets_.array()-min)/(max-min);
    offset += n_basis;
  }
  
  assert(offset == getParameterVectorSize());   
};

void ModelParametersLWR::setParameterVector(const VectorXd& values, bool normalized) {

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
  
  /*
  MatrixXd old_slopes = slopes_;
  for (int i_dim=0; i_dim<n_dims; i_dim++)
  {
    slopes_.col(i_dim) = values.segment(offset,size);
    offset += size;
    // Cache must not be cleared, because normalizedKernelActivations() returns the same values.
  }
  */
  l = "slopes";
  if (isParameterSelected(l)) {
    for (int i_dim=0; i_dim<n_dims; i_dim++)
      slopes_.col(i_dim) = values.segment(offset+i_dim*n_basis,n_basis);
    if (normalized)
      slopes_ = ((max_[l]-min_[l])*slopes_.array())+min_[l];
    offset += n_dims*n_basis;
    clearCache(); // Centers updated, activation need to be updated.
  }
  
  l = "offsets";
  if (isParameterSelected(l)) {
    offsets_ = values.segment(offset,n_basis);
    if (normalized)
      offsets_ = ((max_[l]-min_[l])*slopes_.array())+min_[l];
    offset += n_basis;
    // Cache must not be cleared, because kernelActivations() returns the same values.
  }

  assert(offset == expected_size);   
};


void ModelParametersLWR::setParameterVectorModifierPrivate(std::string modifier, bool new_value)
{
  if (modifier.compare("lines_pivot_at_max_activation")==0)
    set_lines_pivot_at_max_activation(new_value);
  
  if (modifier.compare("slopes_as_angles")==0)
    set_slopes_as_angles(new_value);
  
}

UnifiedModel* ModelParametersLWR::toUnifiedModel(void) const
{

  // LWR uses normalized basis functions
  bool normalized_basis_functions = true;
  return new UnifiedModel(centers_, widths_, slopes_, offsets_, normalized_basis_functions); 
  
}

}
