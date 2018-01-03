/**
 * @file   FunctionApproximatorRLS.cpp
 * @brief  FunctionApproximatorRLS class source file.
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


#include "dmpbbo_io/EigenFileIO.hpp"

#include <iostream>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/LU>

using namespace std;
using namespace Eigen;

namespace DmpBbo {

Eigen::MatrixXd weightedLeastSquares(
  const Eigen::Ref<const Eigen::MatrixXd>& inputs, 
  const Eigen::Ref<const Eigen::MatrixXd>& targets,
  const Eigen::Ref<const Eigen::VectorXd>& weights,
  bool use_offset,
  double regularization,
  double min_weight
  )
{
  assert(inputs.rows() == weights.rows());
  assert(inputs.rows() == targets.rows());
  
  int n_samples = inputs.rows();
  
  cout << "  use_offset=" << use_offset << endl;
  
  // Make the design matrix
  MatrixXd X;
  if (use_offset)
  {
    // Add a column with 1s
    X = MatrixXd::Ones(inputs.rows(),inputs.cols()+1);
    X.leftCols(inputs.cols()) = inputs;
  }
  else
  {
    X = inputs;
  }
  
  int n_betas = X.cols(); 
  cout << "  n_betas=" << n_betas << endl;
  MatrixXd W;
  
  if (min_weight<=0.0)
  {
    // Use all data
    
    // Weights matrix
    W = weights.asDiagonal();
    
    // Regularization matrix
    MatrixXd Gamma = regularization*MatrixXd::Identity(n_betas,n_betas); 
    
    // Compute beta
    // 1 x n_betas 
    // = inv( (n_betas x n_sam)*(n_sam x n_sam)*(n_sam*n_betas) )*( (n_betas x n_sam)*(n_sam x n_sam)*(n_sam * 1) )   
    // = inv(n_betas x n_betas)*(n_betas x 1)
    // Least squares then is a one-liner
    return (X.transpose()*W*X + Gamma).inverse()*X.transpose()*W*targets;
  } 
  else
  {
    // Very low weights do not contribute to the line fitting
    // Therefore, we can delete the rows in W, X and targets for which W is small
    //
    // Example with min_weight = 0.1 (a very high value!! usually it will be lower)
    //    W =       [0.001 0.01 0.5 0.98 0.46 0.01 0.001]^T
    //    X =       [0.0   0.1  0.2 0.3  0.4  0.5  0.6 ; 
    //               1.0   1.0  1.0 1.0  1.0  1.0  1.0  ]^T  (design matrix, so last column = 1)
    //    targets = [1.0   0.5  0.4 0.5  0.6  0.7  0.8  ]
    //
    // will reduce to
    //    W_sub =       [0.5 0.98 0.46 ]^T
    //    X_sub =       [0.2 0.3  0.4 ; 
    //                   1.0 1.0  1.0  ]^T  (design matrix, last column = 1)
    //    targets_sub = [0.4 0.5  0.6  ]
    // 
    // Why all this trouble? Because the submatrices will often be much smaller than the full
    // ones, so they are much faster to invert (note the .inverse() call)
    
    // Get a vector where 1 represents that weights >= min_weight, and 0 otherswise
    VectorXi large_enough = (weights.array() >= min_weight).select(VectorXi::Ones(weights.size()), VectorXi::Zero(weights.size()));

    // Number of samples in the submatrices
    int n_samples_sub = large_enough.sum();
  
    // This would be a 1-liner in Matlab... but Eigen is not good with splicing.
    VectorXd weights_sub(n_samples_sub);
    MatrixXd X_sub(n_samples_sub,n_betas);
    MatrixXd targets_sub(n_samples_sub,targets.cols());
    int jj=0;
    for (int ii=0; ii<n_samples; ii++)
    {
      if (large_enough[ii]==1)
      {
        weights_sub[jj] = weights[ii];
        X_sub.row(jj) = X.row(ii);
        targets_sub.row(jj) = targets.row(ii);
        jj++;
      }
    }
    
    // Do the same inversion as above, but with only a small subset of the data
    
    // Weights matrix
    MatrixXd W_sub = weights_sub.asDiagonal();
    
    // Regularization matrix
    MatrixXd Gamma = regularization*MatrixXd::Identity(n_betas,n_betas); 
    
    // Least squares then is a one-liner
    return (X_sub.transpose()*W_sub*X_sub+Gamma).inverse()*X_sub.transpose()*W_sub*targets_sub;
 
  }
  
}

void linearPrediction(
  const Eigen::Ref<const Eigen::MatrixXd>& inputs, 
  const Eigen::Ref<const Eigen::VectorXd>& beta,
  Eigen::MatrixXd& outputs)
{
  int n_beta = beta.size();
  int n_input_dims = inputs.cols();

  if (n_input_dims==n_beta)
  {
    outputs = inputs*beta;
  }
  else
  {
    assert(n_input_dims==(n_beta-1)); 
    // There is an offset (AKA bias or intercept), so add a column with 1s
    MatrixXd X = MatrixXd::Ones(inputs.rows(),inputs.cols()+1);
    X.leftCols(inputs.cols()) = inputs;
    outputs = X*beta;
  }
      
}

}
