/**
 * @file   FunctionApproximatorLWR.cpp
 * @brief  FunctionApproximatorLWR class source file.
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

#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/BasisFunction.hpp"

#include "eigenutils/eigen_json.hpp"

#include <nlohmann/json.hpp>

#include <iostream>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/LU>



using namespace std;
using namespace Eigen;

namespace DmpBbo {

FunctionApproximatorLWR::FunctionApproximatorLWR(const Eigen::MatrixXd& centers, const Eigen::MatrixXd& widths, const Eigen::MatrixXd& slopes, const Eigen::MatrixXd& offsets, bool asymmetric_kernels, bool lines_pivot_at_max_activation) 
:
  n_basis_functions_(centers.rows()),
  centers_(centers),
  widths_(widths),
  slopes_(slopes), 
  offsets_(offsets),
  asymmetric_kernels_(asymmetric_kernels),
  lines_pivot_at_max_activation_(lines_pivot_at_max_activation),
  slopes_as_angles_(false)
{
#ifndef NDEBUG // Variables below are only required for asserts; check for NDEBUG to avoid warnings.
  int n_dims = centers.cols();
#endif  
  assert(n_basis_functions_==widths_.rows());
  assert(n_dims            ==widths_.cols());
  assert(n_basis_functions_==slopes_.rows());
  assert(n_dims            ==slopes_.cols());
  assert(n_basis_functions_==offsets_.rows());
  assert(1                 ==offsets_.cols());
  
  lines_one_prealloc_ = MatrixXd(1,n_basis_functions_);
  activations_one_prealloc_ = MatrixXd(1,n_basis_functions_);
  
};


void FunctionApproximatorLWR::predict(const Eigen::Ref<const Eigen::MatrixXd>& inputs, MatrixXd& outputs) const
{
  
  int n_time_steps = inputs.rows();
  if (n_time_steps==1) // Only one sample
  {
    ENTERING_REAL_TIME_CRITICAL_CODE

    // Only 1 sample, so real-time execution is possible. No need to allocate memory.
    getLines(inputs, lines_one_prealloc_);

    // Weight the values for each line with the normalized basis function activations  
    bool normalize_activations = true;
    BasisFunction::Gaussian::activations(
      centers_,widths_,inputs,
      activations_one_prealloc_,
      normalize_activations,asymmetric_kernels_);
  
    outputs = (lines_one_prealloc_.array()*activations_one_prealloc_.array()).rowwise().sum();  
    
    EXITING_REAL_TIME_CRITICAL_CODE
    
  }
  else
  {
    
    // The next two lines are not real-time, as they allocate memory
    MatrixXd lines(n_time_steps,n_basis_functions_);
    MatrixXd activations(n_time_steps,n_basis_functions_);
    
    getLines(inputs, lines);

    // Weight the values for each line with the normalized basis function activations  
    bool normalize_activations = true;
    BasisFunction::Gaussian::activations(
      centers_,widths_,inputs,
      activations,
      normalize_activations,asymmetric_kernels_);
  
    outputs = (lines.array()*activations.array()).rowwise().sum();  
    
  }
  
}

void FunctionApproximatorLWR::set_lines_pivot_at_max_activation(bool lines_pivot_at_max_activation)
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

void FunctionApproximatorLWR::set_slopes_as_angles(bool slopes_as_angles)
{
  slopes_as_angles_ = slopes_as_angles;
  cerr << __FILE__ << ":" << __LINE__ << ":";
  cerr << "Not implemented yet!!!" << endl;
  slopes_as_angles_ = false;
}

/*
The code below was previously implemted as follows. Below is the real-time version.


void FunctionApproximatorLWR::getLines(const Eigen::Ref<const Eigen::MatrixXd>& inputs, MatrixXd& lines) const
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

void FunctionApproximatorLWR::getLines(const Eigen::Ref<const Eigen::MatrixXd>& inputs, MatrixXd& lines) const
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
void FunctionApproximatorLWR::kernelActivationsSymmetric(const MatrixXd& centers, const MatrixXd& widths, const Eigen::Ref<const Eigen::MatrixXd>& inputs, MatrixXd& kernel_activations)
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

void from_json(const nlohmann::json& j, FunctionApproximatorLWR*& obj) {
  nlohmann::json jm = j.at("_model_params");
  MatrixXd centers = jm.at("centers").at("values");
  MatrixXd widths = jm.at("widths").at("values");
  MatrixXd slopes = jm.at("slopes").at("values");
  MatrixXd offsets = jm.at("offsets").at("values");  
  obj = new FunctionApproximatorLWR(centers,widths,slopes,offsets);
}

void FunctionApproximatorLWR::to_json_helper(nlohmann::json& j) const {
  j["centers_"] = centers_;
  j["widths_"] = widths_;
  j["offsets_"] = offsets_;
  j["slopes_"] = slopes_;
  j["py/object"] = "dynamicalsystems.FunctionApproximatorLWR.FunctionApproximatorLWR"; // jsonpickle
}

}
