/**
 * @file   getFunctionApproximatorByName.cpp
 * @brief  getFunctionApproximatorByName class source file.
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

#include "functionapproximators/getFunctionApproximatorByName.hpp"

#include "functionapproximators/FunctionApproximator.hpp"

#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/MetaParametersLWPR.hpp"
#include "functionapproximators/FunctionApproximatorLWPR.hpp"
#include "functionapproximators/MetaParametersGMR.hpp"
#include "functionapproximators/FunctionApproximatorGMR.hpp"
#include "functionapproximators/MetaParametersRBFN.hpp"
#include "functionapproximators/FunctionApproximatorRBFN.hpp"
#include "functionapproximators/MetaParametersIRFRLS.hpp"
#include "functionapproximators/FunctionApproximatorIRFRLS.hpp"
#include "functionapproximators/MetaParametersGPR.hpp"
#include "functionapproximators/FunctionApproximatorGPR.hpp"

#include <eigen3/Eigen/Core>

using namespace Eigen;
using namespace std;

namespace DmpBbo {

FunctionApproximator* getFunctionApproximatorByName(std::string name, int n_input_dims)
{
  int n_args = 1;
  
  char** args = new char * [1];
  args[0] = new char [name.length()+1];
  std::strcpy(args[0], name.c_str());  
  
  return getFunctionApproximatorFromArgs(n_args, args, n_input_dims);
}

FunctionApproximator* getFunctionApproximatorFromArgs(int n_args, char* args[], int n_input_dims)
{
  string fa_name = "LWR";
  for (int aa=1; aa<n_args; aa++)
    if (string(args[aa]).compare("fa")==0)                 
      fa_name = string(args[++aa]);

  if (fa_name.compare("LWR")==0)
  {
    //___________________________________________________________________________
    // Locally Weighted Regression
    int n_basis_functions = 11;
    double intersection = 0.5;
    for (int aa=1; aa<n_args; aa++)
    {
      if (string(args[aa]).compare("n_basis_functions")==0)  n_basis_functions = stoi(args[++aa]);
      if (string(args[aa]).compare("intersection")==0)       intersection = stod(args[++aa]);
    }
    
    MetaParametersLWR* pars = new MetaParametersLWR(n_input_dims,n_basis_functions,intersection);
    return new FunctionApproximatorLWR(pars);
    
  }
  else if (fa_name.compare("RBFN")==0)
  {
    //___________________________________________________________________________
    // Radial Basis Function Network
    int n_basis_functions = 9;
    double intersection = 0.5;
    for (int aa=1; aa<n_args; aa++)
    {
      if (string(args[aa]).compare("n_basis_functions")==0)  n_basis_functions = stoi(args[++aa]);
      if (string(args[aa]).compare("intersection")==0)       intersection = stod(args[++aa]);
    }
    VectorXi num_rfs_per_dim = VectorXi::Constant(n_input_dims,n_basis_functions);
    MetaParametersRBFN* pars = new MetaParametersRBFN(n_input_dims,num_rfs_per_dim,intersection);
    return new FunctionApproximatorRBFN(pars);
  }
  else if (fa_name.compare("LWPR")==0)
  {
    // Locally Weighted Projection Regression
#ifdef USE_LWPR
    double   w_gen=0.2;
    double   w_prune=0.8;
    bool     update_D=true;
    double   init_alpha=0.1;
    double   penalty=0.005;
    VectorXd init_D=VectorXd::Constant(n_input_dims,200);
    for (int aa=1; aa<n_args; aa++)
    {
      if (string(args[aa]).compare("w_gen")==0)      w_gen = stoi(args[++aa]);
      if (string(args[aa]).compare("w_prune")==0)    w_prune = stod(args[++aa]);
      if (string(args[aa]).compare("update_D")==0)   update_D = stoi(args[++aa]);
      if (string(args[aa]).compare("init_alpha")==0) init_alpha = stod(args[++aa]);
      if (string(args[aa]).compare("penalty")==0)    penalty = stod(args[++aa]);
      if (string(args[aa]).compare("init_d")==0) 
        init_D=VectorXd::Constant(n_input_dims,stod(args[++aa]));
    }
    MetaParametersLWPR* pars = new MetaParametersLWPR(n_input_dims,init_D,w_gen,w_prune,update_D,init_alpha,penalty);
    return new FunctionApproximatorLWPR(pars);
#else
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "Sorry, LWPR is not available. Is it installed? Returning NULL." << endl;
    return NULL;
#endif // USE_LWPR
  }

  if (fa_name.compare("GMR")==0)
  {
    // Gaussian Mixture Regression  
    int n_basis_functions = 10;
    for (int aa=1; aa<n_args; aa++)
    {
      if (string(args[aa]).compare("n_basis_functions")==0)  n_basis_functions = stoi(args[++aa]);
    }

    MetaParametersGMR* pars = new MetaParametersGMR(n_input_dims,n_basis_functions);
    return new FunctionApproximatorGMR(pars);
  }
  
  if (fa_name.compare("IRFRLS")==0)
  {
    // IRFRLS
    int n_basis_functions = 20;
    double lambda=0.2;
    double gamma=5;
    for (int aa=1; aa<n_args; aa++)
    {
      if (string(args[aa]).compare("n_basis_functions")==0)  n_basis_functions = stoi(args[++aa]);
      if (string(args[aa]).compare("lambda")==0)  lambda = stod(args[++aa]);
      if (string(args[aa]).compare("gamma")==0)  gamma = stod(args[++aa]);
    }
    MetaParametersIRFRLS* pars = new MetaParametersIRFRLS(n_input_dims,n_basis_functions,lambda,gamma);
    return new FunctionApproximatorIRFRLS(pars);
  }

  if (fa_name.compare("GPR")==0)
  {
    // Gaussian Process Regression
    double maximum_covariance = 1.1*1.1;
    double length = 0.1;
    for (int aa=1; aa<n_args; aa++)
    {
      if (string(args[aa]).compare("maximum_covariance")==0)  maximum_covariance = stod(args[++aa]);
      if (string(args[aa]).compare("length")==0)  length = stod(args[++aa]);
    }
    MetaParametersGPR* pars = new MetaParametersGPR(n_input_dims,maximum_covariance,length);
    return new FunctionApproximatorGPR(pars);
  }

  else
  {
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "Unknown function approximator name '" << fa_name << "'" << endl;
    return NULL;
  }
}

}
