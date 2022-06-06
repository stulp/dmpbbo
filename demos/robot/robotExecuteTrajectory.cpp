/**
 * \file robotExecuteTrajectory.cpp
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
       << " <dmp filename.xml> <output trajectory filename.txt>" << endl;
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

  string traj_filename = string(args[1]);
  string cost_vars_filename = string(args[2]);

  cout << "C++    |     Reading traj from file '" << traj_filename << "'"
       << endl;

  // Load trajectory
  Trajectory trajectory;
  trajectory = Trajectory::readFromFile(traj_filename);

  // Prepare simulation
  VectorXd ts = trajectory.ts();
  MatrixXd y_des = trajectory.ys();
  MatrixXd yd_des = trajectory.yds();
  MatrixXd ydd_des = trajectory.ydds();
  int n_time_steps = trajectory.length();
  MatrixXd cost_vars = MatrixXd(n_time_steps, 1 + 6 * 2 + 1);

  // Run simulation
  ThrowBallSimulator simulator;
  double dt = ts[1] - ts[0];
  for (int ii = 0; ii < n_time_steps; ii++) {
    if (ii >= 1) dt = ts[ii] - ts[ii - 1];
    simulator.integrateStep(dt, y_des.row(ii), yd_des.row(ii), ydd_des.row(ii));
    cost_vars.row(ii) = simulator.getState();
  }

  // Save cost_vars to file
  bool overwrite = true;
  cout << "C++ Writing   -> " << cost_vars_filename << endl;
  saveMatrix("./", cost_vars_filename, cost_vars, overwrite);

  return 0;
}
