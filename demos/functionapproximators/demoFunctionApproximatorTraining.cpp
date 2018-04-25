/**
 * \file demoFunctionApproximatorTraining.cpp
 * \author Freek Stulp
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
#include <ctime>
#include <boost/filesystem.hpp>

#include "functionapproximators/FunctionApproximator.hpp"
#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/MetaParametersGMR.hpp"
#include "functionapproximators/FunctionApproximatorGMR.hpp"
#include "functionapproximators/MetaParametersRRRFF.hpp"
#include "functionapproximators/FunctionApproximatorRRRFF.hpp"
#include "functionapproximators/MetaParametersGPR.hpp"
#include "functionapproximators/FunctionApproximatorGPR.hpp"
#include "functionapproximators/MetaParametersRBFN.hpp"
#include "functionapproximators/FunctionApproximatorRBFN.hpp"
#ifdef USE_LWPR
#include "functionapproximators/MetaParametersLWPR.hpp"
#include "functionapproximators/FunctionApproximatorLWPR.hpp"
#endif // USE_LWPR

//#include "getFunctionApproximatorsVector.hpp"
#include "targetFunction.hpp"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;


FunctionApproximator* getFunctionApproximatorByName(std::string name, int input_dim)
{
  
  FunctionApproximator* fa = NULL;
  if (name.compare("LWR")==0)
  {
    // Locally Weighted Regression
    double intersection = 0.6;
    Eigen::VectorXi num_rfs_per_dim = Eigen::VectorXi::Constant(input_dim,9);
    if (input_dim==2) {
      num_rfs_per_dim[0] = 7;
      num_rfs_per_dim[1] = 5;
    }
    double regularization = 0.0;
    MetaParametersLWR* meta_params = new MetaParametersLWR(input_dim,num_rfs_per_dim,intersection,regularization);
    fa = new FunctionApproximatorLWR(meta_params);
  }
  else if (name.compare("GMR")==0)
  {
    // Gaussian Mixture Regression  
    int number_of_gaussians = 10;
    if (input_dim==2) number_of_gaussians = 10;
    MetaParametersGMR* meta_params = new MetaParametersGMR(input_dim,number_of_gaussians);
    fa = new FunctionApproximatorGMR(meta_params);
  }
  else if (name.compare("RRRFF")==0)
  {
    // RRRFF
    int number_of_basis_functions=18;
    if (input_dim==2) number_of_basis_functions = 100;
    double regularization=0.2;
    double gamma=5;
    MetaParametersRRRFF* meta_params = new MetaParametersRRRFF(input_dim,number_of_basis_functions,regularization,gamma);
    fa = new FunctionApproximatorRRRFF(meta_params);
  }
  else if (name.compare("RBFN")==0)
  {
    // Radial Basis Function Network
    double intersection = 0.7;
    int n_rfs = 9;
    if (input_dim==2) n_rfs = 5;
    Eigen::VectorXi num_rfs_per_dim = Eigen::VectorXi::Constant(input_dim,n_rfs);
    double regularization = 0.0;
    MetaParametersRBFN* meta_params = new MetaParametersRBFN(input_dim,num_rfs_per_dim,intersection,regularization);
    fa = new FunctionApproximatorRBFN(meta_params);
  }
  else if (name.compare("GPR")==0)
  {
    // Gaussian Process Regression
    double maximum_covariance = 1.1*1.1;
    double length = 0.1;
    if (input_dim==2) 
    {
      maximum_covariance = 0.1*0.1;
      length = 0.2;
    }
    MetaParametersGPR* meta_params = new MetaParametersGPR(input_dim,maximum_covariance,length);
    fa = new FunctionApproximatorGPR(meta_params);
  }
  else if (name.compare("LWPR")==0)
  {
    // Locally Weighted Projection Regression
#ifdef USE_LWPR
    double   w_gen=0.2;
    double   w_prune=0.8;
    bool     update_D=true;
    double   init_alpha=0.1;
    double   penalty=0.005;
    Eigen::VectorXd init_D=Eigen::VectorXd::Constant(input_dim,200);
    MetaParametersLWPR* meta_params = new MetaParametersLWPR(input_dim,init_D,w_gen,w_prune,update_D,init_alpha,penalty);
    fa = new FunctionApproximatorLWPR(dynamic_cast<MetaParametersLWPR*>(meta_params));
#else
    std::cerr << __FILE__ << ":" << __LINE__ << ":";
    std::cerr << "Sorry, LWPR is not available. Is it installed? Returning NULL." << std::endl;
    return NULL;
#endif // USE_LWPR
  }


  
  if (fa==NULL)
  {
    std::cerr << __FILE__ << ":" << __LINE__ << ":";
    std::cerr << "Function approximator with name '" << name << "' is unknown. Returning NULL." << std::endl;
  }
  
  return fa;
}

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
  cout << "   \t(range of target data is " << mat1.colwise().maxCoeff().array()-mat1.colwise().minCoeff().array() << ")" << endl;
  
  return mean_abs_error_per_output_dim;
}


int main(int n_args, char** args)
{
  // First argument may be optional directory to write data to
  string directory;
  if (n_args>1)
    directory = string(args[1]);
  
  vector<string> fa_names;
  if (n_args>2)
  {
    for (int aa=2; aa<n_args; aa++)
      fa_names.push_back(string(args[aa]));
  }
  else
  {
    fa_names.push_back("LWR");
    fa_names.push_back("RRRFF");
    fa_names.push_back("LWPR");
    fa_names.push_back("GMR");
  }
        
  
  for (int n_input_dims=1; n_input_dims<=2; n_input_dims++)
  {
    
    // Generate training data 
    VectorXi n_samples_per_dim = VectorXi::Constant(1,30); // 25
    if (n_input_dims==2) 
      n_samples_per_dim = VectorXi::Constant(2,10); // 2,25
    
    MatrixXd inputs, targets, outputs;
    targetFunction(n_samples_per_dim,inputs,targets);
    // Add some noise
    //targets = targets + 0.1*VectorXd::Random(targets.rows(),targets.cols());
    
    
    for (unsigned int i_name=0; i_name<fa_names.size(); i_name++)
    {
      
      FunctionApproximator* cur_fa = getFunctionApproximatorByName(fa_names[i_name],n_input_dims);
      if (cur_fa==NULL)
        continue;
  
      string save_directory;
      if (!directory.empty())
        save_directory = directory+"/"+fa_names[i_name]+"_"+(n_input_dims==1?"1D":"2D");
      
      bool overwrite = true;
      cout << "Training " << cur_fa->getName() << " on " << n_input_dims << "D data...\t";
      cur_fa->train(inputs,targets,save_directory,overwrite);

      
      // Do predictions and compute mean absolute error
      outputs.resize(inputs.rows(),cur_fa->getExpectedOutputDim());
      cur_fa->predict(inputs,outputs);

      meanAbsoluteErrorPerOutputDimension(targets,outputs);
      
#ifdef NDEBUG
      // Here, we time the predict function on single inputs, i.e. typical usage in a
      // real-time loop on a robot.
      
      // It's better to test this code with -03 instead of -ggdb. That's why its only run 
      // when NDEBUG is set.

      MatrixXd input = inputs.row(0);
      MatrixXd output(input.rows(),input.cols());
      int n_calls = 1000000;
      clock_t begin = clock();
      for (int call=0; call<n_calls; call++)
      {
        cur_fa->predict(input,output);
        input(0,0) += 0.1/n_calls;
      }
      clock_t end = clock();
      double time_sec = (double)(end - begin) / static_cast<double>( CLOCKS_PER_SEC );
      cout << "  time for " << n_calls << " calls of predict: "<< time_sec;
      double time_per_call = time_sec/n_calls; 
      cout << " => " << (int)(1.0/(time_per_call*1000)) << "kHz" << endl;
#endif
      
      delete cur_fa;

      
    }
  
  }
  
  return 0;
}


