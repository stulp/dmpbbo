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

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <string>

#include "functionapproximators/FunctionApproximator.hpp"

#include "eigenutils/eigen_realtime_check.hpp"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;
using namespace nlohmann;

int main(int n_args, char** args)
{
  string directory = "../../../../python/functionapproximators/tests/";

  for (int n_dims : {1, 2}) {
    for (string fa_name : {"RBFN", "LWR"}) {
      string filename = directory + fa_name + "_" + to_string(n_dims) + "D.json";
  
      cout << "========================================================" << endl;
      cout << filename << endl;
  
      ifstream file(filename);
      if (file.fail()) {
        cerr << "File not found: " << filename << endl;
        continue;
      }
      json j = json::parse(file);
      cout << j << endl;
  
      cout << "from_json ===============" << endl;
      FunctionApproximator* fa = j.get<FunctionApproximator*>();
      
      cout << "<< ===============" << endl;
      cout << *fa << endl;
  
      cout << "to_json ===============" << endl;
      json j2 = fa;
      cout << j2 << endl;
      FunctionApproximator* fa2 = j2.get<FunctionApproximator*>();
  
      
      cout << "real-time ===============" << endl;
      // Here, we time the predict function on single inputs, i.e. typical usage
      // in a real-time loop on a robot. We check if memory is allocated with
      // ENTERING_REAL_TIME_CRITICAL_CODE
      
      MatrixXd input = MatrixXd::Ones(1, n_dims);
      MatrixXd output(1, 1);
      ENTERING_REAL_TIME_CRITICAL_CODE
      for (int ii = 0; ii < 3; ii++) {
        fa->predict(input, output);
        fa2->predict(input, output);
      }
      EXITING_REAL_TIME_CRITICAL_CODE
      
     delete fa;
    }
  }

  return 0;
}
