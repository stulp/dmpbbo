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
  ifstream file("../demos/cpp/json/Dmp_for_cpp.json");
  Dmp* dmp = json::parse(file).get<Dmp*>();

  VectorXd x(dmp->dim(), 1);
  VectorXd xd(dmp->dim(), 1);
  VectorXd y(dmp->dim_y(), 1);
  VectorXd yd(dmp->dim_y(), 1);
  VectorXd ydd(dmp->dim_y(), 1);

  dmp->integrateStart(x, xd);
  double dt = 0.001;
  for (double t = 0.0; t < 2.0; t+=dt) {
    dmp->integrateStep(dt, x, x, xd);
    // Convert complete DMP state to end-eff state
    dmp->stateAsPosVelAcc(x, xd, y, yd, ydd);
  }
}
