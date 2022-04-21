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

#define EIGEN_RUNTIME_NO_MALLOC  // Enable runtime tests for allocations

#include <eigen3/Eigen/Core>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

#include "dynamicalsystems/DynamicalSystem.hpp"
#include "eigenutils/eigen_file_io.hpp"

using namespace std;
using namespace DmpBbo;
using namespace Eigen;
using namespace nlohmann;

int main(int n_args, char** args)
{
  if (n_args != 3) {
    cout << "Usage: " << args[0] << " <directory> <basename>" << endl;
    return -1;
  }

  string directory = args[1];
  string basename = args[2];

  cout << "========================================================" << endl;
  string filename_json = directory + "/" + basename + ".json";
  cout << filename_json << endl;
  ifstream file(filename_json);
  if (file.fail()) {
    cerr << "Could not find: " << filename_json << endl;
    return -1;
  }
  json j = json::parse(file);
  DynamicalSystem* d = j.get<DynamicalSystem*>();

  cout << j << endl;
  cout << *d << endl;

  VectorXd ts;
  string filename_ts = directory + "/ts.txt";
  if (!loadMatrix(filename_ts, ts)) {
    cerr << "Could not find: " << filename_ts << endl;
    return -1;
  }

  // Prepare analytical solution
  MatrixXd xs, xds;

  cout << "===============" << endl << "C++ Analytical solution" << endl;
  d->analyticalSolution(ts, xs, xds);
  bool overwrite = true;
  saveMatrix(directory + "/xs_analytical.txt", xs, overwrite);
  saveMatrix(directory + "/xds_analytical.txt", xds, overwrite);

  // Numerical integration
  VectorXd x(d->dim(), 1);
  VectorXd xd(d->dim(), 1);
  double dt;

  cout << "===============" << endl << "C++ Integrating with Euler" << endl;
  d->integrateStart(x, xd);
  xs.row(0) = x;
  xds.row(0) = xd;
  int n_time_steps = ts.size();
  for (int t = 1; t < n_time_steps; t++) {
    dt = ts[t] - ts[t - 1];
    d->integrateStepEuler(dt, xs.row(t - 1), x, xd);
    xs.row(t) = x;
    xds.row(t) = xd;
  }
  saveMatrix(directory + "/xs_euler.txt", xs, overwrite);
  saveMatrix(directory + "/xds_euler.txt", xds, overwrite);

  cout << "===============" << endl
       << "C++ Integrating with Runge-Kutta" << endl;
  d->integrateStart(x, xd);
  xs.row(0) = x;
  xds.row(0) = xd;
  for (int t = 1; t < n_time_steps; t++) {
    dt = ts[t] - ts[t - 1];
    d->integrateStepRungeKutta(dt, xs.row(t - 1), x, xd);
    xs.row(t) = x;
    xds.row(t) = xd;
  }
  saveMatrix(directory + "/xs_rungekutta.txt", xs, overwrite);
  saveMatrix(directory + "/xds_rungekutta.txt", xds, overwrite);

  return 0;
}
