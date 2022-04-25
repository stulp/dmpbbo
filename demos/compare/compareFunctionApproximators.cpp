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

#include "eigenutils/eigen_file_io.hpp"
#include "functionapproximators/FunctionApproximator.hpp"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;
using namespace nlohmann;

int main(int n_args, char** args)
{
  if (n_args != 4) {
    cout << "Usage: " << args[0] << " <directory> <fa_name> <n_dims>" << endl;
    return -1;
  }

  string directory = args[1];
  string fa_name = args[2];
  int n_dims = atoi(args[3]);
  string basename = fa_name + "_" + to_string(n_dims) + "D";

  cout << "========================================================" << endl;
  string filename_json = directory + "/" + basename + "_for_cpp.json";
  cout << filename_json << endl;
  ifstream file(filename_json);
  if (file.fail()) {
    cerr << "Could not find: " << filename_json << endl;
    return -1;
  }
  json j = json::parse(file);
  FunctionApproximator* fa = j.get<FunctionApproximator*>();

  cout << j << endl;
  cout << *fa << endl;

  MatrixXd inputs;
  string filename_inputs = directory + "/" + basename + "_inputs.txt";
  if (!loadMatrix(filename_inputs, inputs)) {
    cerr << "Could not find: " << filename_inputs << endl;
    return -1;
  }

  cout << "===============" << endl << "C++ Analytical solution" << endl;
  MatrixXd outputs;
  fa->predict(inputs, outputs);
  bool overwrite = true;
  saveMatrix(directory + "/" + basename + "_outputs.txt", outputs, overwrite);

  delete fa;

  return 0;
}
