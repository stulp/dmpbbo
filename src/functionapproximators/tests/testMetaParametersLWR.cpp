/**
 * \file testMetaParametersLWR.cpp
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

#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/ModelParametersLWR.hpp"
//#include "functionapproximators/FunctionApproximatorLWR.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"
#include "../demos/targetFunction.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <eigen3/Eigen/Core>
#include <boost/filesystem.hpp>

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

int main(int n_args, char** args)
{
  string directory;
  double intersection = 0.5;
  int n_dims = 1;
  
  if (n_args>1)
    directory = string(args[1]);
  if (n_args>2)
    n_dims = atoi(args[2]);
  if (n_args>3)
    intersection = atof(args[3]);
    
  
  // Generete the activations of the basis functions and save to file
  VectorXi n_samples_per_dim = VectorXi::Constant(1,200);
  if (n_dims==2) 
    n_samples_per_dim = VectorXi::Constant(2,20);
    
  MatrixXd inputs, targets; // Not really interested in targets, but useful to get inputs
  targetFunction(n_samples_per_dim,inputs,targets);
  
  VectorXd min = inputs.colwise().minCoeff();
  VectorXd max = inputs.colwise().maxCoeff();
  
    
  MatrixXd centers, widths;
  MetaParametersLWR* mp;
  if (n_dims==1)
  {
    // Try constructor for 1D input data.
    int n_bfs = 9;
    mp = new MetaParametersLWR(n_dims,n_bfs,intersection); 
  }
  else
  {
    // Try constructor for N-D input data.
    
    // First constructor for N-D input data: specify only number of basis functions per dimension
    VectorXi n_bfs_per_dim(n_dims);
    n_bfs_per_dim[0] = 3;
    n_bfs_per_dim[1] = 5;
    mp = new MetaParametersLWR(n_dims,n_bfs_per_dim,intersection);
    cout << *mp << endl;
    delete mp; // Free memory to try other constructor
    
    // Second constructor for N-D input data: specify centers of the basis functions per dimension
    vector<VectorXd> centers_per_dim(n_dims);
    for (int i_dim=0; i_dim<n_dims; i_dim++)
      centers_per_dim[i_dim] = VectorXd::LinSpaced(n_bfs_per_dim[i_dim],min[i_dim],max[i_dim]);
     mp = new MetaParametersLWR(n_dims,centers_per_dim,intersection); 
  }
  
  
  cout << *mp << endl;
  mp->getCentersAndWidths(min,max,centers,widths);
  cout << "centers = " << centers << endl;
  cout << "widths  = " << widths << endl;
  delete mp;
}

