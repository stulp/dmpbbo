/**
 * \file demoLWRTraining.cpp
 * \author Freek Stulp
 *
 * This file is part of DmpBbo, a set of libraries and programs for the 
 * black-box optimization of dynamical movement primitives.
 * Copyright (C) 2018 Freek Stulp
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

#include "dmpbbo_io/EigenFileIO.hpp"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;


int main(int n_args, char** args)
{
  double intersection = 0.5;
  double n_rfs = 9;
  string directory = string(args[1]);
  if (n_args>2)
    intersection = atof(args[2]);
  if (n_args>3)
    n_rfs = atoi(args[3]);
  
  // Load training data 
  MatrixXd inputs;
  MatrixXd targets;
  directory += "/";
  if (!loadMatrix(directory+"inputs.txt", inputs)) return -1;
  if (!loadMatrix(directory+"targets.txt", targets)) return -1;
  int input_dim = inputs.cols();

  // Initialize function approximator
  MetaParametersLWR* meta_params = new MetaParametersLWR(input_dim,n_rfs,intersection);
  FunctionApproximator* fa = new FunctionApproximatorLWR(meta_params);

  // Train function approximator with data
  bool overwrite = true;
  fa->train(inputs,targets,directory,overwrite);

  // Make predictions for the targets
  MatrixXd outputs(inputs.rows(),fa->getExpectedOutputDim());
  fa->predict(inputs,outputs);
  saveMatrix(directory,"outputs.txt",outputs,overwrite);

  VectorXd min(1); min << 0.0;
  VectorXd max(1); max << 2.0;
  VectorXi n_samples_grid(1); n_samples_grid << 200;
  fa->saveGridData(min, max, n_samples_grid, directory, overwrite);

  delete fa;
  
  return 0;
}


