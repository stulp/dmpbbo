/**
 * @file   BasisFunction.cpp
 * @brief  BasisFunction class source file.
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
 

#include "BasisFunction.hpp"

#include <iostream>

#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/LU>

using namespace Eigen;
using namespace std;

namespace DmpBbo {

namespace BasisFunction {

void Gaussian::activations(
    const std::vector<Eigen::VectorXd>& mus, 
    const std::vector<Eigen::MatrixXd>& covars, 
    std::vector<double> priors, 
    const Eigen::MatrixXd& inputs, 
    Eigen::MatrixXd& kernel_activations, 
    bool normalized_basis_functions)
{

  unsigned int n_basis_functions = mus.size();
  int n_samples         = inputs.rows();

  assert(n_basis_functions>0); 
  assert(n_basis_functions==covars.size());
#ifndef NDEBUG // Variables below are only required for asserts; check for NDEBUG to avoid warnings.
  int n_dims = mus[0].size();
#endif  
  assert(n_dims==covars[0].cols());
  assert(n_dims==covars[0].rows());
  assert(n_dims==inputs.cols());

  kernel_activations.resize(n_samples,n_basis_functions);  

  if (normalized_basis_functions && n_basis_functions==1)
  {
    // Locally Weighted Regression with only one basis function is pretty odd.
    // Essentially, you are taking the "Locally Weighted" part out of the regression, and it becomes
    // standard least squares 
    // Anyhow, for those that still want to "abuse" LWR as R (i.e. without LW), we explicitly
    // set the normalized kernels to 1 here, to avoid numerical issues in the normalization below.
    // (normalizing a Gaussian basis function with itself leads to 1 everywhere).
    kernel_activations.fill(1.0);
    return;
  }  

  VectorXd mu,diff,exp_term;
  MatrixXd covar_inv;
  double prior;
  
  for (unsigned int bb=0; bb<n_basis_functions; bb++)
  {
    mu = mus[bb];
    covar_inv = covars[bb].inverse();
    prior = priors[bb];
    for (int tt=0; tt<n_samples; tt++)
    {
      // Here, we compute the values of a (unnormalized) multi-variate Gaussian:
      //   activation = exp(-0.5*(x-mu)*Sigma^-1*(x-mu))
      diff =  inputs.row(tt)-mu.transpose();
      exp_term = -0.5*diff.transpose()*covar_inv*diff;
      assert(exp_term.size()==1);
      kernel_activations(tt,bb) = prior*exp(exp_term[0]);
    }
  }
  
  if (normalized_basis_functions)
  {
    // Compute sum for each row (each value in input_vector)
    MatrixXd sum_kernel_activations = kernel_activations.rowwise().sum(); // n_samples x 1
  
    // Add small number to avoid division by zero. Not full-proof...  
    if ((sum_kernel_activations.array()==0).any())
      sum_kernel_activations.array() += sum_kernel_activations.maxCoeff()/100000.0;
    
    // Normalize for each row (each value in input_vector)  
    kernel_activations = kernel_activations.array()/sum_kernel_activations.replicate(1,n_basis_functions).array();
  }
  
  
}

void Gaussian::activations(const Eigen::MatrixXd& centers, const Eigen::MatrixXd& widths, const Eigen::MatrixXd& inputs, Eigen::MatrixXd& kernel_activations, bool normalized_basis_functions, bool asymmetric_kernels)
{
  
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

  if (normalized_basis_functions && n_basis_functions==1)
  {
    // Locally Weighted Regression with only one basis function is pretty odd.
    // Essentially, you are taking the "Locally Weighted" part out of the regression, and it becomes
    // standard least squares 
    // Anyhow, for those that still want to "abuse" LWR as R (i.e. without LW), we explicitly
    // set the normalized kernels to 1 here, to avoid numerical issues in the normalization below.
    // (normalizing a Gaussian basis function with itself leads to 1 everywhere).
    kernel_activations.fill(1.0);
    return;
  }  
  
  double c,w,x;
  for (int bb=0; bb<n_basis_functions; bb++)
  {

    // Here, we compute the values of a (unnormalized) multi-variate Gaussian:
    //   activation = exp(-0.5*(x-mu)*Sigma^-1*(x-mu))
    // Because Sigma is diagonal in our case, this simplifies to
    //   activation = exp(\sum_d=1^D [-0.5*(x_d-mu_d)^2/Sigma_(d,d)]) 
    //              = \prod_d=1^D exp(-0.5*(x_d-mu_d)^2/Sigma_(d,d)) 
    // This last product is what we compute below incrementally
    
    kernel_activations.col(bb).fill(1.0);
    for (int i_dim=0; i_dim<n_dims; i_dim++)
    {
      c = centers(bb,i_dim);
      for (int i_s=0; i_s<n_samples; i_s++)
      {
        x = inputs(i_s,i_dim);
        w = widths(bb,i_dim);
        
        if (asymmetric_kernels && x<c && bb>0)
          // Get the width of the previous basis function
          // This is the part that makes it assymetric
          w = widths(bb-1,i_dim);
          
        kernel_activations(i_s,bb) *= exp(-0.5*pow(x-c,2)/(w*w));
      }
    }
  }

  if (normalized_basis_functions)
  {
    // Compute sum for each row (each value in input_vector)
    MatrixXd sum_kernel_activations = kernel_activations.rowwise().sum(); // n_samples x 1
  
    // Add small number to avoid division by zero. Not full-proof...  
    if ((sum_kernel_activations.array()==0).any())
      sum_kernel_activations.array() += sum_kernel_activations.maxCoeff()/100000.0;
    
    // Normalize for each row (each value in input_vector)  
    kernel_activations = kernel_activations.array()/sum_kernel_activations.replicate(1,n_basis_functions).array();
  }

}

void Cosine::activations(
    const std::vector<Eigen::MatrixXd>& angular_frequencies,
    const std::vector<Eigen::VectorXd>& phases,
    const Eigen::MatrixXd& inputs, 
    Eigen::MatrixXd& activations)
{
  unsigned int n_basis_functions = angular_frequencies.size();
  int n_samples                  = inputs.rows();
  
  assert(n_basis_functions>0);
  assert(phases.size()==n_basis_functions);
  assert(phases[0].size()==1);
  // input_cols is input dim
  assert(angular_frequencies[0].size()==inputs.cols());

  
  activations.resize(n_samples,n_basis_functions);  
  
  for (unsigned int bb=0; bb<n_basis_functions; bb++)
  {
    for (int i_s=0; i_s<n_samples; i_s++)
    {
      activations(i_s,bb) = cos(angular_frequencies[bb].row(0).dot(inputs.row(i_s)) + phases[bb][0]);
    }
  }
  
}

void Cosine::activations(
  const Eigen::MatrixXd& angular_frequencies,
  const Eigen::VectorXd& phases,
  const Eigen::MatrixXd& inputs, 
  Eigen::MatrixXd& activations)
{
  // Activations for each basis function are computed with:
  //   activation(bb) = cos(inputs(bb)*freqs(bb).transpose() + phase(bb)) 
  activations = inputs * angular_frequencies.transpose();
  activations.rowwise() += phases.transpose();
  activations = activations.array().cos();
}


} // namespace BasisFunction

} // namespace DmpBbo


