/**
 * \file testSaveGMMMAtrix.cpp
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

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <time.h>
#include <boost/filesystem.hpp>

#include "functionapproximators/ModelParametersGMR.hpp"
#include "functionapproximators/MetaParametersGMR.hpp"
#include "functionapproximators/FunctionApproximatorGMR.hpp"

#include "../demos/targetFunction.hpp"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

int main(int n_args, char** args)
{
  string directory("/tmp/testSaveGMMMatrix/");
  if (n_args>1)
    directory = string(args[1]);
  
  for (int n_input_dims=1; n_input_dims<=2; n_input_dims++)
  {
    VectorXi n_samples_per_dim = VectorXi::Constant(1,30);
    if (n_input_dims==2) 
      n_samples_per_dim = VectorXi::Constant(2,10);
    
    MatrixXd inputs, targets, outputs;
    targetFunction(n_samples_per_dim,inputs,targets);
    
    
    int number_of_gaussians = 3;
    if (n_input_dims==2) number_of_gaussians = 4;    
    FunctionApproximator* fa = new FunctionApproximatorGMR(new MetaParametersGMR(n_input_dims,number_of_gaussians));
  
    string save_directory;
    if (!directory.empty())
      save_directory = directory+"/GMR_"+(n_input_dims==1?"1D":"2D");
    
    bool overwrite = true;
    cout << "Training on " << n_input_dims << "D data...\n";
    fa->train(inputs,targets,save_directory,overwrite);
    
    const ModelParametersGMR* gmm = dynamic_cast<const ModelParametersGMR*>(fa->getModelParameters());
    
    string filename = save_directory+"/gmm.txt";
    cout << "Saving GMM to file " << filename << "\n";
    gmm->saveGMMToMatrix(filename,overwrite);
      
    
    ModelParametersGMR* gmm_new = ModelParametersGMR::loadGMMFromMatrix(filename);
    FunctionApproximatorGMR* fa_new = new FunctionApproximatorGMR(gmm_new);
    cout << *fa_new << endl;

    MatrixXd gmm_matrix;
    gmm->toMatrix(gmm_matrix);     
    cout << gmm_matrix << endl;
    cout << "____________________" << endl;
    
    if (gmm_new==NULL)
    {
      cout << "gmm_new is NULL...." << endl;
    }
    else
    {
      MatrixXd gmm_new_matrix;
      gmm_new->toMatrix(gmm_new_matrix);
      cout << gmm_matrix << endl;
    }
    
  }

  return 0;
}


