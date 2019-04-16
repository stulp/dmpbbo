/**
 * \file getFunctionApproximatorsVector.hpp
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

#ifndef GETFUNCTIONAPPROXIMATORSVECTOR_H
#define GETFUNCTIONAPPROXIMATORSVECTOR_H

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

#include <vector>
#include <iostream>
#include <eigen3/Eigen/Core>

namespace DmpBbo {
  
class MetaParameters;
class FunctionApproximator;

FunctionApproximator* getFunctionApproximatorByName(std::string name, int input_dim);

void getFunctionApproximatorsVector(int input_dim, std::vector<FunctionApproximator*>& function_approximators);


void getFunctionApproximatorsVector(int input_dim, std::vector<FunctionApproximator*>& function_approximators)
{
  std::vector<std::string> names;
  names.push_back("LWR");
  names.push_back("LWPR");
  names.push_back("GMR");
  names.push_back("RRRFF");
  names.push_back("RBFN");
  names.push_back("GPR");

  for (unsigned int i_name=0; i_name<names.size(); i_name++)
  {
    FunctionApproximator* fa = getFunctionApproximatorByName(names[i_name],input_dim);
    if (fa!=NULL)
      function_approximators.push_back(fa);
  }
}

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

}

#endif        //  #ifndef GETFUNCTIONAPPROXIMATORSVECTOR_H

