/**
 * \author Freek Stulp
 *
 * This file is part of DmpBbo, a set of libraries and programs for the 
 * black-box optimization of dynamical movement primitives.
 * Copyright (C) 2022 Freek Stulp
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
#include <fstream>
#include <string>
#include <set>

#include <nlohmann/json.hpp>

#include "dynamicalsystems/DynamicalSystem.hpp"
#include "dynamicalsystems/DynamicalSystemFactory.hpp"



using namespace std;
using namespace DmpBbo;
using namespace nlohmann;

int main(int n_args, char** args)
{
  
  string directory = "../../../../python/dynamicalsystems/tests/";
  string filename = "ExponentialSystem_1D.json";  
  if (n_args>1)
    filename = string(args[1]);
  filename = directory + filename;  
  
  
  ifstream file(filename);
  json j = json::parse(file);
  cout << j << endl;

  
  DynamicalSystem* dyn_sys = NULL;
  DynamicalSystemFactory::from_jsonpickle(j,dyn_sys);
  cout << *dyn_sys << endl;
  
  return 0;
}


