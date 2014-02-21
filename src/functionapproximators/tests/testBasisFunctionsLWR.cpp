/**
 * \file testBasisFunctionsLWR.cpp
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
#include <string>
#include <fstream>

#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/ModelParametersLWR.hpp"
#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "../demos/targetFunction.hpp"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

int main(int n_args, char** args)
{
  string directory;
  if (n_args>1)
    directory = string(args[1]);
  //else
  //  usage(args[0],"/tmp/testFunctionApproximatorLWR");

  for (int input_dim=1; input_dim<=1; input_dim++)
  {
    VectorXi n_samples_per_dim = VectorXi::Constant(1,50);
    if (input_dim==2) 
      n_samples_per_dim = VectorXi::Constant(2,20);
    
    MatrixXd inputs, targets, outputs;
    targetFunction(n_samples_per_dim,inputs,targets);


    int n_rfs = 9;
    if (input_dim==2) 
      n_rfs = 3;

    // Random returns values in range [-1 1]
    VectorXd centers(n_rfs);
    centers << 0.0,0.1,0.2,0.3,0.4,1.0,1.2,1.5,2.2; 
    //centers = VectorXd::LinSpaced(n_rfs,0,2);
    vector<VectorXd> centers_per_dim;
    centers_per_dim.push_back(centers);
    
    double intersection_height = 0.56;

    string save_directory;
    bool overwrite = true;
    
    for (int asymmetric_kernels=0; asymmetric_kernels<=1; asymmetric_kernels++)
    {
      if (!directory.empty())
        save_directory = directory+(input_dim==1?"/1D":"/2D")+(asymmetric_kernels==1?"_Asymmetric":"_symmetric");
      
      MetaParametersLWR* meta_parameters = new MetaParametersLWR(input_dim,centers_per_dim,intersection_height,asymmetric_kernels==1);
   
      FunctionApproximator* fa = new FunctionApproximatorLWR(meta_parameters);
      fa->train(inputs,targets,save_directory,overwrite);
      
      delete fa;
    }
    
  }
  
  
  
  
  return 0;
}

