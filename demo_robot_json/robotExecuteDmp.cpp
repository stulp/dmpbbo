/**
 * \file robotExecuteDmp.cpp
 * \author Freek Stulp
 *
 * \ingroup Demos
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

#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "dmp/Dmp.hpp"
#include "dmp/Trajectory.hpp"
#include "eigenutils/eigen_file_io.hpp"
#include "runSimulationThrowBall.hpp"

using namespace nlohmann;
using namespace std;
using namespace Eigen;
using namespace DmpBbo;

void help(char* binary_name)
{
  cout << "Usage: " << binary_name
       << " <dmp filename.json> <cost vars filename.txt>" << endl;
}

/** Main function
 * \param[in] n_args Number of arguments
 * \param[in] args Arguments themselves
 * \return Success of exection. 0 if successful.
 */
int main(int n_args, char** args)
{
  if (n_args != 3) {
    help(args[0]);
    return -1;
  }

  if (string(args[1]).compare("--help") == 0) {
    help(args[0]);
    return 0;
  }

  string dmp_filename = string(args[1]);
  string cost_vars_filename = string(args[2]);

  cout << "C++ Reading <-   " << dmp_filename << endl;

  ifstream file(dmp_filename);
  json j = json::parse(file);

  Dmp* dmp = j.get<Dmp*>();

  // Integrate DMP longer than the tau with which it was trained
  double integration_time = 1.5 * dmp->tau();
  double frequency_Hz = 100.0;
  int n_time_steps = floor(frequency_Hz * integration_time);
  VectorXd ts =
      VectorXd::LinSpaced(n_time_steps, 0, integration_time);  // Time steps

  Trajectory trajectory;
  dmp->analyticalSolution(ts, trajectory);

  MatrixXd cost_vars;
  runSimulationThrowBall(&trajectory, cost_vars);

  bool overwrite = true;
  cout << "C++ Writing   -> " << cost_vars_filename << endl;
  saveMatrix("./", cost_vars_filename, cost_vars, overwrite);

  delete dmp;

  return 0;
}
