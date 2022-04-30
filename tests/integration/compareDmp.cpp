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

#include "dmp/Dmp.hpp"
#include "eigenutils/eigen_file_io.hpp"
#include "functionapproximators/FunctionApproximator.hpp"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;
using namespace nlohmann;

int main(int n_args, char** args)
{
  if (n_args != 3) {
    cout << "Usage: " << args[0] << " <directory> <dmp_name>" << endl;
    return -1;
  }

  string directory = args[1];
  string dmp_name = args[2];

  cout << "========================================================" << endl;
  string filename_json = directory + "/" + dmp_name + "_for_cpp.json";
  cout << filename_json << endl;
  ifstream file(filename_json);
  if (file.fail()) {
    cerr << "Could not find: " << filename_json << endl;
    return -1;
  }
  json j = json::parse(file);
  Dmp* dmp = j.get<Dmp*>();

  cout << j << endl;
  cout << *dmp << endl;

  VectorXd ts;
  if (!loadMatrix(directory + "/ts.txt", ts)) return -1;

  cout << "===============" << endl << "C++ Analytical solution";

  MatrixXd xs, xds, forcing_terms, fa_outputs;
  dmp->analyticalSolution(ts, xs, xds, forcing_terms, fa_outputs);

  cout << " (saving)" << endl;
  bool overwrite = true;
  saveMatrix(directory, "xs_ana.txt", xs, overwrite);
  saveMatrix(directory, "xds_ana.txt", xds, overwrite);
  saveMatrix(directory, "forcing_terms_ana.txt", forcing_terms, overwrite);
  saveMatrix(directory, "fa_outputs_ana.txt", fa_outputs, overwrite);

  cout << "===============" << endl << "C++ Numerical integration";

  VectorXd x(dmp->dim(), 1);
  VectorXd xd(dmp->dim(), 1);
  dmp->integrateStart(x, xd);
  xs.row(0) = x;
  xds.row(0) = xd;
  int n_time_steps = ts.size();
  for (int t = 1; t < n_time_steps; t++) {
    double dt = ts[t] - ts[t - 1];
    dmp->integrateStepRungeKutta(dt, xs.row(t - 1), x, xd);
    xs.row(t) = x;
    xds.row(t) = xd;
  }

  cout << " (saving)" << endl;
  saveMatrix(directory + "/xs_step.txt", xs, overwrite);
  saveMatrix(directory + "/xds_step.txt", xds, overwrite);

  delete dmp;

  return 0;
}
