/**
 * \file testFunctionApproximatorCloning.cpp
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

#include "getFunctionApproximatorsVector.hpp"

#include "functionapproximators/FunctionApproximator.hpp"
#include "../demos/targetFunction.hpp"

#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;
using namespace DmpBbo;


int main(int n_args, char** args)
{
  int n_input_dims = 1;
  vector<FunctionApproximator*> function_approximators;
  if (n_args==1)
  {
    // No name passed, get all function approximators
    getFunctionApproximatorsVector(n_input_dims,function_approximators);
  }
  else
  {
    // Assume the arguments are names of function approximatores
    for (int i_arg=1; i_arg<n_args; i_arg++)
    {
      FunctionApproximator* fa =  getFunctionApproximatorByName(args[i_arg],n_input_dims);
      if (fa==NULL)
        return -1;
      function_approximators.push_back(fa);
    }
  }

  VectorXi n_samples_per_dim = VectorXi::Constant(n_input_dims,50);
  MatrixXd inputs, targets;
  targetFunction(n_samples_per_dim, inputs, targets);
  
  for (unsigned int dd=0; dd<function_approximators.size(); dd++)
  {
    
    FunctionApproximator* cur_fa = function_approximators[dd]; 
    FunctionApproximator* cloned = cur_fa->clone(); 
    
    cout << endl <<  endl << "__________________________________________________________________________" << endl;
    
    cout << endl << "CLONED" << endl;
    cout << "Original   :" << endl << "    " << *cur_fa << endl;
    cout << "Clone      :" << endl << "    " << *cloned << endl;

    cout << endl << "TRAINING CLONE" << endl;
    cloned->train(inputs,targets);
    cout << "Original   :" << endl << "    " << *cur_fa << endl;
    cout << "Clone      :" << endl << "    " << *cloned << endl;

    cout << endl << "CLONE OF TRAINED CLONE" << endl;
    FunctionApproximator* cloned_cloned = cloned->clone(); 
    cout << "Original   :" << endl << "    " << *cur_fa << endl;
    cout << "Clone      :" << endl << "    " << *cloned << endl;
    cout << "Clone clone:" << endl << "    " << *cloned_cloned << endl;

    cout << endl << "DELETING CLONE" << endl;
    delete cloned;
    cout << "Original   :" << endl << "    " << *cur_fa << endl;
    cout << "Clone clone:" << endl << "    " << *cloned_cloned << endl;

    cout << endl << "DELETING ORIGINAL" << endl;
    delete cur_fa;
    cout << "Clone clone:" << endl << "    " << *cloned_cloned << endl;
    
    delete cloned_cloned;
    
  }
}

