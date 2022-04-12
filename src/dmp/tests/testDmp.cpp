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
#include <set>
#include <string>

#include "dmp/Dmp.hpp"
#include "dmp/Trajectory.hpp"
#include "eigenutils/eigen_file_io.hpp"
#include "eigenutils/eigen_realtime_check.hpp"

using namespace std;
using namespace Eigen;
using namespace nlohmann;
using namespace DmpBbo;

int main(int n_args, char** args)
{
  string directory = "";
  if (n_args > 1) directory = string(args[1]);

  // Test JSON
  string filename_dmp = "../../../../python/dmp/tests/Dmp.json";
  cout << "* Reading and parsing: " << filename_dmp << endl;
  ifstream file(filename_dmp);
  json j = json::parse(file);
  cout << j << endl;

  Dmp* dmp = j.get<Dmp*>();
  cout << *dmp << endl;

  // Test real-time integration
  VectorXd x(dmp->dim(), 1);
  VectorXd x_updated(dmp->dim(), 1);
  VectorXd xd(dmp->dim(), 1);

  double dt = 0.01;
  for (string integration_method : {"Euler", "Runge-Kutta", "Default"}) {
    cout << "* Integrating real-time with: " << integration_method << endl;
    dmp->integrateStart(x, xd);

    ENTERING_REAL_TIME_CRITICAL_CODE
    if (integration_method == "Euler") {
      for (int t = 0; t < 3; t++) dmp->integrateStepEuler(dt, x, x_updated, xd);
    } else if (integration_method == "Runge-Kutta") {
      for (int t = 0; t < 3; t++)
        dmp->integrateStepRungeKutta(dt, x, x_updated, xd);
    } else {
      for (int t = 0; t < 3; t++) dmp->integrateStep(dt, x, x_updated, xd);
    }
    EXITING_REAL_TIME_CRITICAL_CODE
  }

  bool loading_succesful = false;
  VectorXd ts;
  if (boost::filesystem::exists(directory + "/ts.txt"))
    loading_succesful = loadMatrix(directory + "/ts.txt", ts);
  if (!loading_succesful) ts = VectorXd::LinSpaced(76, 0, 0.75);

  cout << "* Analytical solution." << endl;
  MatrixXd xs_ana, xds_ana, forcing_terms, fa_output;
  dmp->analyticalSolution(ts, xs_ana, xds_ana, forcing_terms, fa_output);
  Trajectory traj_reproduced;
  dmp->statesAsTrajectory(ts, xs_ana, xds_ana, traj_reproduced);

  cout << "* Integrating step-by-step." << endl;
  // VectorXd x(dmp->dim(), 1);
  // VectorXd xd(dmp->dim(), 1);
  // VectorXd x_updated(dmp->dim(), 1);
  dmp->integrateStart(x, xd);

  MatrixXd xs_step(ts.size(), x.size());
  MatrixXd xds_step(ts.size(), xd.size());
  xs_step.row(0) = x;
  xds_step.row(0) = xd;

  for (int t = 1; t < ts.size(); t++) {
    double dt = ts[t] - ts[t - 1];
    dmp->integrateStep(dt, x, x_updated, xd);
    x = x_updated;
    xs_step.row(t) = x;
    xds_step.row(t) = xd;
  }

  if (loading_succesful) {
    cout << "* Saving to directory: " << directory << endl;

    MatrixXd output_ana(ts.size(), 1 + xs_ana.cols() + xds_ana.cols());
    output_ana << ts, xs_ana, xds_ana;
    bool overwrite = true;
    saveMatrix(directory, "ts_xs_xds_ana.txt", output_ana, overwrite);
    saveMatrix(directory, "forcing_terms_ana.txt", forcing_terms, overwrite);
    saveMatrix(directory, "fa_output_ana.txt", fa_output, overwrite);

    traj_reproduced.saveToFile(directory, "traj_reproduced_ana.txt", overwrite);

    MatrixXd output_step(ts.size(), 1 + xs_step.cols() + xds_step.cols());
    output_step << ts, xs_step, xds_step;
    saveMatrix(directory, "ts_xs_xds_step.txt", output_step, overwrite);

    // MatrixXd tau_mat(1, 1);
    // tau_mat(0, 0) = dmp->tau();
    // saveMatrix(directory, "tau.txt", tau_mat, overwrite);
  }

  return 0;
}
