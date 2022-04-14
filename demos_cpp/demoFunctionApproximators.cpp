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

#include "eigenutils/eigen_realtime_check.hpp"
#include "functionapproximators/FunctionApproximator.hpp"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;
using namespace nlohmann;

int main(int n_args, char** args)
{
  string directory = "../../demos_cpp/json/";

  for (int n_dims : {1, 2}) {
    for (string fa_name : {"RBFN", "LWR"}) {
      string label = fa_name + "_" + to_string(n_dims) + "D";
      string filename = directory + label + ".json";

      cout << "======================================================" << endl;
      cout << filename << endl;

      ifstream file(filename);
      if (file.fail()) {
        cerr << "File not found: " << filename << endl;
        continue;
      }
      json j = json::parse(file);
      cout << j << endl;

      cout << "from_json " + label + " ===============" << endl;
      FunctionApproximator* fa = j.get<FunctionApproximator*>();

      cout << "<< ===============" << endl;
      cout << *fa << endl;

      cout << "to_json " + label + " ===============" << endl;
      json j2 = fa;
      cout << j2 << endl;
      FunctionApproximator* fa2 = j2.get<FunctionApproximator*>();

      cout << "predict " + label + " ===============" << endl;
      int n_samples = 10;
      MatrixXd inputs_mat = MatrixXd::Ones(n_samples, n_dims);
      MatrixXd outputs_mat(n_samples, 1);
      fa->predict(inputs_mat, outputs_mat);
      fa2->predict(inputs_mat, outputs_mat);

      cout << "predict real-time " + label + " ===============" << endl;
      RowVectorXd input_vec = RowVectorXd::Ones(n_dims);
      VectorXd output_vec(1);
      ENTERING_REAL_TIME_CRITICAL_CODE
      for (int ii = 0; ii < 3; ii++) {
        fa->predictRealTime(input_vec, output_vec);
        fa2->predictRealTime(input_vec, output_vec);
      }
      EXITING_REAL_TIME_CRITICAL_CODE

      delete fa;
    }
  }

  return 0;
}
