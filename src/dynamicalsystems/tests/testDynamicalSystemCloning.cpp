/**
 * @file testDynamicalSystemCloning.cpp
 * @author Freek Stulp
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

#include "getDynamicalSystemsVector.hpp"

#include "dynamicalsystems/DynamicalSystem.hpp"

#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;
using namespace DmpBbo;  

int main(int n_args, char** args)
{
  vector<DynamicalSystem*> dyn_systems;
  getDynamicalSystemsVector(dyn_systems);
  
  for (unsigned int dd=0; dd<dyn_systems.size(); dd++)
  {
    DynamicalSystem* cur_dyn_system = dyn_systems[dd]; 
    DynamicalSystem* cloned = dyn_systems[dd]->clone(); 
    
    cout << "Original:" << endl << "    " << *cur_dyn_system << endl;
    
    // Delete current dynamical system to see if it doesn't delete memory in the clone
    delete cur_dyn_system;
    
    cloned->set_initial_state(cloned->initial_state());
    cloned->set_tau(cloned->tau());
    cout << "   Clone:" << endl << "    " << *cloned << endl;
    delete cloned;
    
  }
}

