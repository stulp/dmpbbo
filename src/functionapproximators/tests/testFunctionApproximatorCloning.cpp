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
#include "testTargetFunction.hpp"

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
  getFunctionApproximatorsVector(n_input_dims,function_approximators);
  
  for (unsigned int dd=0; dd<function_approximators.size(); dd++)
  {
    
    FunctionApproximator* cur_fa = function_approximators[dd]; 
    FunctionApproximator* cloned = cur_fa->clone(); 
    
    cout << endl <<  endl << "__________________________________________________________________________" << endl;
    
    cout << endl << "CLONED" << endl;
    cout << "Original   :" << endl << "    " << *cur_fa << endl;
    cout << "Clone      :" << endl << "    " << *cloned << endl;


    cout << endl << "DELETING CLONE" << endl;
    delete cloned;
    cout << "Original   :" << endl << "    " << *cur_fa << endl;

    cout << endl << "DELETING ORIGINAL" << endl;
    delete cur_fa;
    
  }
}

