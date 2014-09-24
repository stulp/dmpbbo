/**
 * \file testToUnifiedModel.cpp
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

#include "functionapproximators/UnifiedModel.hpp"

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
  string directory, directory_fa;
  if (n_args>1)
    directory = string(args[1]);
  bool overwrite = true;
  
  for (int n_input_dims = 1; n_input_dims<=2; n_input_dims++)
  {
    vector<FunctionApproximator*> function_approximators;
    if (n_args>2)
    {
      // Assume the arguments are names of function approximatores
      for (int i_arg=2; i_arg<n_args; i_arg++)
      {
        FunctionApproximator* fa =  getFunctionApproximatorByName(args[i_arg],n_input_dims);
        if (fa==NULL)
          return -1;
        function_approximators.push_back(fa);
      }
    }
    else
    {
      // No name passed, get all function approximators
      getFunctionApproximatorsVector(n_input_dims,function_approximators);
    }
  
    // Generate training data 
    VectorXi n_samples_per_dim = VectorXi::Constant(1,25);
    if (n_input_dims==2) 
      n_samples_per_dim = VectorXi::Constant(2,10);
      
    MatrixXd inputs, targets, outputs;
    targetFunction(n_samples_per_dim,inputs,targets);
  
    VectorXd min = inputs.colwise().minCoeff();
    VectorXd max = inputs.colwise().maxCoeff();
        
    VectorXi n_samples_per_dim_dense = VectorXi::Constant(n_input_dims,100);
    if (n_input_dims==2)
      n_samples_per_dim = VectorXi::Constant(n_input_dims,40);
            
    
    
    for (unsigned int dd=0; dd<function_approximators.size(); dd++)
    {
      
      FunctionApproximator* fa = function_approximators[dd]; 
    
      cout << "_____________________________________" << endl << fa->getName() << endl;
      cout << "    Training (with " << n_input_dims << "D data)"<< endl;
      if (!directory.empty()) {
        directory_fa =  directory+"/"+fa->getName();
        if (n_input_dims==1)
          directory_fa = directory_fa+"1D";
        else
          directory_fa = directory_fa+"2D";
      }
        
      fa->train(inputs,targets,directory_fa,overwrite);
      fa->predict(inputs,outputs);
      
      cout << "    Converting to UnifiedModel"  << endl;
      UnifiedModel* mp_unified =  fa->getUnifiedModel();
      if (mp_unified!=NULL)
      {
        mp_unified->saveGridData(min, max, n_samples_per_dim_dense, directory_fa+"Unified",overwrite);
      
        fa->saveGridData(min, max, n_samples_per_dim_dense, directory_fa,overwrite);
      
        saveMatrix(directory_fa+"Unified","inputs.txt",inputs,overwrite);
        saveMatrix(directory_fa+"Unified","targets.txt",targets,overwrite);
        //saveMatrix(directory_fa+"Unified","outputs.txt",outputs,overwrite);
      }
      
      delete fa;
      fa = NULL;
      delete mp_unified;
    }
  }
     
  return 0;
}


