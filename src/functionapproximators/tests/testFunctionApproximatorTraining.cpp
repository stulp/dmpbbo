/**
 * \file testFunctionApproximatorTraining.cpp
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
#include <time.h>
#include <boost/filesystem.hpp>

#include "functionapproximators/FunctionApproximator.hpp"

#include "getFunctionApproximatorsVector.hpp"
#include "../demos/targetFunction.hpp"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

int main(int n_args, char** args)
{
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
    fa_names.push_back("IRFRLS");
    fa_names.push_back("LWPR");
    fa_names.push_back("GMR");
  }
        
  
  for (int n_input_dims=1; n_input_dims<=2; n_input_dims++)
  {
    
    VectorXi n_samples_per_dim = VectorXi::Constant(1,30);
    if (n_input_dims==2) 
      n_samples_per_dim = VectorXi::Constant(2,25);
    
    MatrixXd inputs, targets, outputs;
    targetFunction(n_samples_per_dim,inputs,targets);
    
    
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
      cur_fa->predict(inputs,outputs);

      MatrixXd abs_error = (targets.array()-outputs.array()).abs();
      VectorXd mean_abs_error_per_output_dim = abs_error.colwise().mean();
     
      cout << fixed << setprecision(5);
      cout << "         Mean absolute error ";
      if (mean_abs_error_per_output_dim.size()>1) cout << " (per dimension)";
      cout << ": " << mean_abs_error_per_output_dim.transpose();      
      cout << "   \t(range of target data is " << targets.colwise().maxCoeff().array()-targets.colwise().minCoeff().array() << ")";
      
      cout << endl;
      
      delete cur_fa;
    }
  
  }
  
  return 0;
}


