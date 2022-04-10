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
#include "functionapproximators/MetaParametersRBFN.hpp"
#include "functionapproximators/FunctionApproximatorRBFN.hpp"

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
  names.push_back("RBFN");

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

  
  if (fa==NULL)
  {
    std::cerr << __FILE__ << ":" << __LINE__ << ":";
    std::cerr << "Function approximator with name '" << name << "' is unknown. Returning NULL." << std::endl;
  }
  
  return fa;
}

}

#endif        //  #ifndef GETFUNCTIONAPPROXIMATORSVECTOR_H

