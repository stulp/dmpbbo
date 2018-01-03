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
#include <ctime>

#include "dmpbbo_io/EigenFileIO.hpp"

#include "functionapproximators/LeastSquares.hpp"

#include "../demos/targetFunction.hpp"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

int main(int n_args, char** args)
{
  string directory;
  if (n_args>1)
    directory = string(args[1]);
  
  for (int n_input_dims=1; n_input_dims<=2; n_input_dims++)
  {
    // Prepare data
    MatrixXd inputs, targets, outputs;
    if (n_input_dims==2) 
    {
      VectorXi n_samples_per_dim = VectorXi::Constant(2,10);
      targetFunction(n_samples_per_dim,inputs,targets);
    }
    else
    {
      int n_samples = 30;
      inputs.resize(n_samples,1);
      targets.resize(n_samples,1);
      inputs = VectorXd::LinSpaced(n_samples,0.0,2.0);
      targets = 2.0*inputs.array() + 3.0;
      targets += 0.25*MatrixXd::Random(n_samples,1);
    }

    // Parameters for least squares 
    int n_samples = inputs.rows();
    VectorXd weights = VectorXd::Ones(n_samples);
    bool use_offset = true;
    double regularization = 0.1;
    double min_weight = 0.0;
      
    cout << "Least squares on " << n_input_dims << "D data...\t";
    VectorXd beta = weightedLeastSquares(inputs,targets,weights,use_offset,regularization,min_weight);

    cout << "  beta=" << beta << endl;
    linearPrediction(inputs,beta,outputs);
    
    // Prepare directory
    string save_directory;
    if (!directory.empty())
      save_directory = directory+"/"+(n_input_dims==1?"1D":"2D");
    bool overwrite = true;
    
    saveMatrix(save_directory,"inputs.txt",inputs,overwrite);
    saveMatrix(save_directory,"weights.txt",weights,overwrite);
    saveMatrix(save_directory,"targets.txt",targets,overwrite);
    VectorXd tmp = VectorXd::Constant(1,regularization);
    saveMatrix(save_directory,"regularization.txt",tmp,overwrite);
    saveMatrix(save_directory,"beta.txt",beta,overwrite);
    saveMatrix(save_directory,"outputs.txt",outputs,overwrite);
    
    cout << endl;
  }
  
  return 0;
}


