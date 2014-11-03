/**
 * \file demoTrainFunctionApproximators.cpp
 * \author Freek Stulp
 * \brief  Demonstrates how to initialize and train a function approximator..
 *
 * \ingroup Demos
 * \ingroup FunctionApproximators
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

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <time.h>
#include <boost/filesystem.hpp>

#include "functionapproximators/FunctionApproximatorGMR.hpp"
#include "functionapproximators/FunctionApproximatorIRFRLS.hpp"
#include "functionapproximators/FunctionApproximatorLWPR.hpp"
#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/FunctionApproximatorRBFN.hpp"
#include "functionapproximators/FunctionApproximatorGPR.hpp"
#include "functionapproximators/MetaParametersGMR.hpp"
#include "functionapproximators/MetaParametersIRFRLS.hpp"
#include "functionapproximators/MetaParametersLWPR.hpp"
#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/MetaParametersRBFN.hpp"
#include "functionapproximators/MetaParametersGPR.hpp"


#include "targetFunction.hpp"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

/** Compute mean absolute error for each column of two matrices.
 *  \param[in] mat1 First matrix of data 
 *  \param[in] mat2 Second matrix of data
 *  \return Mean absolute error between the matrices (one value for each column)
 */
VectorXd meanAbsoluteErrorPerOutputDimension(const MatrixXd& mat1, const MatrixXd& mat2)
{
  MatrixXd abs_error = (mat1.array()-mat2.array()).abs();
  VectorXd mean_abs_error_per_output_dim = abs_error.colwise().mean();
     
  cout << fixed << setprecision(5);
  cout << "         Mean absolute error ";
  if (mean_abs_error_per_output_dim.size()>1) cout << " (per dimension)";
  cout << ": " << mean_abs_error_per_output_dim.transpose();      
  cout << "   \t(range of target data is " << mat1.colwise().maxCoeff().array()-mat1.colwise().minCoeff().array() << ")";
  
  return mean_abs_error_per_output_dim;
}

/** Main function
 * \param[in] n_args Number of arguments
 * \param[in] args Arguments themselves
 * \return Success of exection. 0 if successful.
 */
int main(int n_args, char** args)
{
  // First argument may be optional directory to write data to
  string directory, directory_fa;
  if (n_args>1)
    directory = string(args[1]);
  bool overwrite = true;

  
  
  // Generate training data 
  int n_input_dims = 1;
  VectorXi n_samples_per_dim = VectorXi::Constant(1,25);
  if (n_input_dims==2) 
    n_samples_per_dim = VectorXi::Constant(2,25);
    
  MatrixXd inputs, targets, outputs;
  targetFunction(n_samples_per_dim,inputs,targets);
  
  // Add some noise
  //targets = targets + 0.1*VectorXd::Random(targets.rows(),targets.cols());

  FunctionApproximator* fa;
  
  // Locally Weighted Regression
  double intersection = 0.5;
  int n_rfs = 9;
  if (n_input_dims==2) n_rfs = 5;
  VectorXi num_rfs_per_dim = VectorXi::Constant(n_input_dims,n_rfs);
  MetaParametersLWR* meta_parameters_lwr = new MetaParametersLWR(n_input_dims,num_rfs_per_dim,intersection);
  fa = new FunctionApproximatorLWR(meta_parameters_lwr);

  cout << "_____________________________________" << endl << fa->getName() << endl;
  cout << "    Training"  << endl;
  if (!directory.empty()) directory_fa =  directory+"/"+fa->getName();
  fa->train(inputs,targets,directory_fa,overwrite);
  cout << "    Predicting" << endl;
  fa->predict(inputs,outputs);
  meanAbsoluteErrorPerOutputDimension(targets,outputs);
  cout << endl << endl;
  
  delete fa;
  delete meta_parameters_lwr;
    
  // Weighted Least Squares
  // LWR with only one basis function
  vector<VectorXd> centers_per_dim(1);
  // Choose one center between min and max of inputs
  centers_per_dim[0] = 0.5*(inputs.colwise().minCoeff()+inputs.colwise().maxCoeff());
  meta_parameters_lwr = new MetaParametersLWR(n_input_dims, centers_per_dim);

  fa = new FunctionApproximatorLWR(meta_parameters_lwr);

  cout << "_____________________________________" << endl << fa->getName() << endl;
  cout << "    Training"  << endl;
  if (!directory.empty()) directory_fa =  directory+"/WLS";
  fa->train(inputs,targets,directory_fa,overwrite);
  cout << "    Predicting" << endl;
  fa->predict(inputs,outputs);
  meanAbsoluteErrorPerOutputDimension(targets,outputs);
  cout << endl << endl;
  
  delete fa;
  delete meta_parameters_lwr;

  
  // IRFRLS
  int number_of_basis_functions=100;
  double lambda=0.2;
  double gamma=10;
  MetaParametersIRFRLS* meta_parameters_irfrls = new MetaParametersIRFRLS(n_input_dims,number_of_basis_functions,lambda,gamma);
  fa = new FunctionApproximatorIRFRLS(meta_parameters_irfrls);
  
  cout << "_____________________________________" << endl << fa->getName() << endl;
  cout << "    Training"  << endl;
  if (!directory.empty()) directory_fa =  directory+"/"+fa->getName();
  fa->train(inputs,targets,directory_fa,overwrite);
  cout << "    Predicting" << endl;
  fa->predict(inputs,outputs);
  meanAbsoluteErrorPerOutputDimension(targets,outputs);
  cout << endl << endl;
  
  delete fa;
  delete meta_parameters_irfrls;
  
  
  
  // Gaussian Mixture Regression (GMR)
  int number_of_gaussians = pow(5,n_input_dims);
  MetaParametersGMR* meta_parameters_gmr = new MetaParametersGMR(n_input_dims,number_of_gaussians);
  fa = new FunctionApproximatorGMR(meta_parameters_gmr);
    
  cout << "_____________________________________" << endl << fa->getName() << endl;
  cout << "    Training"  << endl;
  if (!directory.empty()) directory_fa =  directory+"/"+fa->getName();
  fa->train(inputs,targets,directory_fa,overwrite);
  cout << "    Predicting" << endl;
  fa->predict(inputs,outputs);
  meanAbsoluteErrorPerOutputDimension(targets,outputs);
  cout << endl << endl;
  
  delete fa;
  delete meta_parameters_gmr;

  
    // Locally Weighted Projection Regression
#ifdef USE_LWPR
    double   w_gen=0.2;
    double   w_prune=0.8;
    bool     update_D=true;
    double   init_alpha=0.1;
    double   penalty=0.005;
    VectorXd init_D=VectorXd::Constant(n_input_dims,200);
  MetaParametersLWPR* meta_parameters_lwpr = new MetaParametersLWPR(n_input_dims,init_D,w_gen,w_prune,update_D,init_alpha,penalty);
  fa = new FunctionApproximatorLWPR(meta_parameters_lwpr);
    
  cout << "_____________________________________" << endl << fa->getName() << endl;
  cout << "    Training"  << endl;
  if (!directory.empty()) directory_fa =  directory+"/"+fa->getName();
  fa->train(inputs,targets,directory_fa,overwrite);
  cout << "    Predicting" << endl;
  fa->predict(inputs,outputs);
  meanAbsoluteErrorPerOutputDimension(targets,outputs);
  cout << endl << endl;
  
  delete fa;
  delete meta_parameters_lwpr;
#endif // USE_LWPR

  // Radial Basis Function Network
  intersection = 0.7;
  n_rfs = 9;
  if (n_input_dims==2) n_rfs = 5;
  num_rfs_per_dim = VectorXi::Constant(n_input_dims,n_rfs);
  MetaParametersRBFN* meta_parameters_rbfn = new MetaParametersRBFN(n_input_dims,num_rfs_per_dim,intersection);
  fa = new FunctionApproximatorRBFN(meta_parameters_rbfn);

  cout << "_____________________________________" << endl << fa->getName() << endl;
  cout << "    Training"  << endl;
  if (!directory.empty()) directory_fa =  directory+"/"+fa->getName();
  fa->train(inputs,targets,directory_fa,overwrite);
  cout << "    Predicting" << endl;
  fa->predict(inputs,outputs);
  meanAbsoluteErrorPerOutputDimension(targets,outputs);
  cout << endl << endl;

  delete fa;
  delete meta_parameters_rbfn;
  
  // Gaussian Process Regression
  double maximum_covariance = 3;
  double length = 0.1;
  MetaParametersGPR* meta_parameters_gpr = new MetaParametersGPR(n_input_dims,maximum_covariance,length);
  fa = new FunctionApproximatorGPR(meta_parameters_gpr);

  cout << "_____________________________________" << endl << fa->getName() << endl;
  cout << "    Training"  << endl;
  if (!directory.empty()) directory_fa =  directory+"/"+fa->getName();
  fa->train(inputs,targets,directory_fa,overwrite);
  cout << "    Predicting" << endl;
  fa->predict(inputs,outputs);
  meanAbsoluteErrorPerOutputDimension(targets,outputs);
  cout << endl << endl;

  delete fa;
  delete meta_parameters_gpr;
 
  return 0;
}


