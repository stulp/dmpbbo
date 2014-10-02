/**
 * \file getFunctionApproximatorsVector.cpp
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

#include "getFunctionApproximatorsVector.hpp"

#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/MetaParametersGMR.hpp"
#include "functionapproximators/FunctionApproximatorGMR.hpp"
#include "functionapproximators/MetaParametersIRFRLS.hpp"
#include "functionapproximators/FunctionApproximatorIRFRLS.hpp"
#include "functionapproximators/MetaParametersGPR.hpp"
#include "functionapproximators/FunctionApproximatorGPR.hpp"
#include "functionapproximators/MetaParametersRBFN.hpp"
#include "functionapproximators/FunctionApproximatorRBFN.hpp"
#ifdef USE_LWPR
#include "functionapproximators/MetaParametersLWPR.hpp"
#include "functionapproximators/FunctionApproximatorLWPR.hpp"
#endif // USE_LWPR

#include <vector>
#include <iostream>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

namespace DmpBbo {

void getFunctionApproximatorsVector(int input_dim, std::vector<FunctionApproximator*>& function_approximators)
{
  vector<string> names;
  names.push_back("LWR");
  names.push_back("LWPR");
  names.push_back("GMR");
  names.push_back("IRFRLS");
  names.push_back("RBFN");
  names.push_back("GPR");

  for (unsigned int i_name=0; i_name<names.size(); i_name++)
  {
    FunctionApproximator* fa = getFunctionApproximatorByName(names[i_name],input_dim);
    if (fa!=NULL)
      function_approximators.push_back(fa);
  }
}

MetaParameters* getMetaParametersByName(string name, int input_dim)
{

  if (name.compare("LWR")==0)
  {
    // Locally Weighted Regression
    double intersection = 0.6;
    VectorXi num_rfs_per_dim = VectorXi::Constant(input_dim,9);
    if (input_dim==2) {
      num_rfs_per_dim[0] = 7;
      num_rfs_per_dim[1] = 5;
    }
    return new MetaParametersLWR(input_dim,num_rfs_per_dim,intersection);
  } 

  if (name.compare("LWPR")==0)
  {
    // Locally Weighted Projection Regression
#ifdef USE_LWPR
    double   w_gen=0.2;
    double   w_prune=0.8;
    bool     update_D=true;
    double   init_alpha=0.1;
    double   penalty=0.005;
    VectorXd init_D=VectorXd::Constant(input_dim,200);
    return new MetaParametersLWPR(input_dim,init_D,w_gen,w_prune,update_D,init_alpha,penalty);
#else
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "Sorry, LWPR is not available. Is it installed? Returning NULL." << endl;
    return NULL;
#endif // USE_LWPR
  }

  if (name.compare("GMR")==0)
  {
    // Gaussian Mixture Regression  
    int number_of_gaussians = 10;
    if (input_dim==2) number_of_gaussians = 10;
    return new MetaParametersGMR(input_dim,number_of_gaussians);
  }
  
  if (name.compare("IRFRLS")==0)
  {
    // IRFRLS
    int number_of_basis_functions=18;
    if (input_dim==2) number_of_basis_functions = 100;
    double lambda=0.2;
    double gamma=5;
    return new MetaParametersIRFRLS(input_dim,number_of_basis_functions,lambda,gamma);
  }


  if (name.compare("RBFN")==0)
  {
    // Radial Basis Function Network
    double intersection = 0.7;
    int n_rfs = 9;
    if (input_dim==2) n_rfs = 5;
    VectorXi num_rfs_per_dim = VectorXi::Constant(input_dim,n_rfs);
    return new MetaParametersRBFN(input_dim,num_rfs_per_dim,intersection);
  }
    
  if (name.compare("GPR")==0)
  {
    // Gaussian Process Regression
    double maximum_covariance = 1.1*1.1;
    double length = 0.1;
    if (input_dim==2) 
    {
      maximum_covariance = 0.1*0.1;
      length = 0.2;
    }
    return new MetaParametersGPR(input_dim,maximum_covariance,length);
  }
  
  cerr << __FILE__ << ":" << __LINE__ << ":";
  cerr << "Meta-parameters with name '" << name << "' is unknown. Returning NULL." << endl;
  return NULL;
}


FunctionApproximator* getFunctionApproximatorByName(string name, int input_dim)
{
  
  MetaParameters* meta_parameters = getMetaParametersByName(name, input_dim);
  FunctionApproximator* fa = NULL;
  if (name.compare("LWR")==0)
    fa = new FunctionApproximatorLWR(dynamic_cast<MetaParametersLWR*>(meta_parameters));
  if (name.compare("GMR")==0)
    fa = new FunctionApproximatorGMR(dynamic_cast<MetaParametersGMR*>(meta_parameters));
  if (name.compare("IRFRLS")==0)
    fa = new FunctionApproximatorIRFRLS(dynamic_cast<MetaParametersIRFRLS*>(meta_parameters));
  if (name.compare("RBFN")==0)
    fa = new FunctionApproximatorRBFN(dynamic_cast<MetaParametersRBFN*>(meta_parameters));
  if (name.compare("GPR")==0)
    fa = new FunctionApproximatorGPR(dynamic_cast<MetaParametersGPR*>(meta_parameters));

  if (name.compare("LWPR")==0)
  {
    // Locally Weighted Projection Regression
#ifdef USE_LWPR
    fa = new FunctionApproximatorLWPR(dynamic_cast<MetaParametersLWPR*>(meta_parameters));
#else
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "Sorry, LWPR is not available. Is it installed? Returning NULL." << endl;
#endif // USE_LWPR
  }


  
  if (fa==NULL)
  {
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "Function approximator with name '" << name << "' is unknown. Returning NULL." << endl;
  }
  
  delete meta_parameters;
  
  return fa;
}

}
