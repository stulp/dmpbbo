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

#include "functionapproximators/FunctionApproximator.hpp"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;
using namespace nlohmann;

int main(int n_args, char** args)
{
  
  string directory = "../../../../python/functionapproximators/tests/";
  
  vector<string> filenames = {"LWR_1D.json","RBFN_1D.json","LWR_2D.json","RBFN_2D.json"};
  
  for (string filename: filenames) {
  
    ifstream file(directory+filename);
    json j = json::parse(file);
    cout << "=================================================================" << endl;
    cout << filename << endl;
    cout << "===============" << endl;   
    cout << j << endl;
    
    FunctionApproximator* fa = j.get<FunctionApproximator*>();
    cout << "===============" << endl;   
    cout << *fa << endl;
  }
  
  return 0;
}


