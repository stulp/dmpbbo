/**
 * \file testToModelParametersUnified.cpp
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

#include "functionapproximators/FunctionApproximatorLWPR.hpp"
#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/FunctionApproximatorRBFN.hpp"
#include "functionapproximators/MetaParametersLWPR.hpp"
#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/MetaParametersRBFN.hpp"
#include "functionapproximators/ModelParametersLWPR.hpp"
#include "functionapproximators/ModelParametersLWR.hpp"
#include "functionapproximators/ModelParametersRBFN.hpp"

#include "functionapproximators/ModelParametersUnified.hpp"

#include "../demos/targetFunction.hpp"
#include "getFunctionApproximatorsVector.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

/** Main function
 * \param[in] n_args Number of arguments
 * \param[in] args Arguments themselves
 * \return Success of exection. 0 if successful.
 */
int main(int n_args, char** args)
{
  // First argument may be optional directory to write data to
  string directory, directory_fa;
  if (n_args>1)
    directory = string(args[1]);
  bool overwrite = true;

  // Generate training data 
  int n_input_dims = 1;
  VectorXi n_samples_per_dim = VectorXi::Constant(1,25);
  if (n_input_dims==2) 
    n_samples_per_dim = VectorXi::Constant(2,25);
    
  MatrixXd inputs, targets, outputs;
  targetFunction(n_samples_per_dim,inputs,targets);

  VectorXd min = inputs.colwise().minCoeff();
  VectorXd max = inputs.colwise().maxCoeff();
      
  VectorXi n_samples_per_dim_dense = VectorXi::Constant(n_input_dims,100);
  if (n_input_dims==2)
    n_samples_per_dim = VectorXi::Constant(n_input_dims,40);
          
  FunctionApproximator* fa = getFunctionApproximatorByName("RBFN", n_input_dims);

  cout << "_____________________________________" << endl << fa->getName() << endl;
  cout << "    Training"  << endl;
  if (!directory.empty()) 
    directory_fa =  directory+"/"+fa->getName();
  fa->train(inputs,targets,directory_fa,overwrite);
  cout << "    Predicting" << endl;
  fa->predict(inputs,outputs);
  cout << endl << endl;
  
  ModelParametersUnified* mp_unified =  fa->getModelParametersUnified();
  //cout << mp_unified->toString() << endl;
  mp_unified->saveGridData(min, max, n_samples_per_dim_dense, directory_fa+"Unified",overwrite);

  const ModelParametersRBFN* mp =  dynamic_cast<const ModelParametersRBFN*>(fa->getModelParameters());
  //cout << mp_unified->toString() << endl;
  mp->saveGridData(min, max, n_samples_per_dim_dense, directory_fa,overwrite);

  saveMatrix(directory_fa+"Unified","inputs.txt",inputs,overwrite);
  saveMatrix(directory_fa+"Unified","targets.txt",targets,overwrite);
  saveMatrix(directory_fa+"Unified","outputs.txt",outputs,overwrite);
  
  delete fa;
  delete mp_unified;
 
  return 0;
}


