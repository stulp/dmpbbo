/**
 * \file demoLeastSquares.cpp
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

// This must be included before any Eigen header files are included
#include "eigen_realtime/eigen_realtime_check.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"

#include "functionapproximators/leastSquares.hpp"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

void process_arguments(int n_args, char** args, double& regularization, bool& use_offset, string& directory);


int main(int n_args, char** args)
{
  string directory;
  bool use_offset = false;
  double regularization = 0.0;
  process_arguments(n_args,args,regularization,use_offset,directory);
  
  // Make inputs and targets, or read them from file (depending on whether directory is passed)
  MatrixXd inputs;
  MatrixXd targets;
  VectorXd weights;
  
  if (directory.empty())
  {
    int n_samples = 25;
    inputs = VectorXd::LinSpaced(n_samples,0.0,2.0);
    targets = 2.0*inputs.array()+ 3.0;
    weights = VectorXd::Ones(n_samples);
  }
  else
  {
    directory += "/";
    if (!loadMatrix(directory+"inputs.txt", inputs)) return -1;
    if (!loadMatrix(directory+"targets.txt", targets)) return -1;
    if (!loadMatrix(directory+"weights.txt", weights)) return -1;
  }
  
  int n_input_dims = inputs.cols();
  cout << "Least squares on " << n_input_dims << "D data ("<< regularization << " " << use_offset << ")\t";
  
  // Call least squares
  VectorXd beta = weightedLeastSquares(inputs,targets,weights,use_offset,regularization);

  cout << "  beta=" << beta.transpose() << endl;

  // Make predictions for the inputs
  MatrixXd outputs;
  ENTERING_REAL_TIME_CRITICAL_CODE
  linearPrediction(inputs,beta,outputs);
  EXITING_REAL_TIME_CRITICAL_CODE
  
  
  bool overwrite = true;
  saveMatrix(directory,"beta.txt",beta,overwrite);
  saveMatrix(directory,"outputs.txt",outputs,overwrite);
  
  cout << endl;
  
  return 0;
}


void process_arguments(int n_args, char** args, double& regularization, bool& use_offset, string& directory)
{
  if (n_args>1)
    regularization = atof(args[1]);
  if (n_args>2)
  {
    int use_offset_int = atoi(args[2]);
    use_offset = (use_offset_int!=0);
  }
  if (n_args>3)
    directory = string(args[3]);
}
