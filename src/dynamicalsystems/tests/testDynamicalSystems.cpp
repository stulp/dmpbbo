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
#include "eigenutils/eigen_realtime_check.hpp"

using namespace std;
using namespace DmpBbo;
using namespace Eigen;
using namespace nlohmann;

int main(int n_args, char** args)
{
  string directory = "../../../../python/dynamicalsystems/tests/";

  vector<string> filenames;
  for (string dim : {"1D", "2D"})
    for (string system : {"Exponential", "Sigmoid", "SpringDamper"})
      filenames.push_back(system + "System_" + dim + ".json");
  filenames.push_back("TimeSystem.json");

  for (string filename : filenames) {
    filename = directory + filename;
    cout << "================================================================="
         << endl;
    cout << filename << endl;

    cout << "===============" << endl;
    ifstream file(filename);
    json j = json::parse(file);
    cout << j << endl;

    cout << "===============" << endl;
    DynamicalSystem* d = j.get<DynamicalSystem*>();
    cout << *d << endl;

    VectorXd x(d->dim(), 1);
    VectorXd x_updated(d->dim(), 1);
    VectorXd xd(d->dim(), 1);

    double dt = 0.01;
    for (string integration_method : {"Euler", "Runge-Kutta", "Default"}) {
      cout << "===============" << endl;
      cout << "Integrating with: " << integration_method << endl;
      d->integrateStart(x, xd);

      ENTERING_REAL_TIME_CRITICAL_CODE
      if (integration_method == "Euler") {
        for (int t = 1; t < 10; t++)
          d->integrateStepEuler(dt, x, x_updated, xd);
      } else if (integration_method == "Runge-Kutta") {
        for (int t = 1; t < 10; t++)
          d->integrateStepRungeKutta(dt, x, x_updated, xd);
      } else {
        for (int t = 1; t < 10; t++) d->integrateStep(dt, x, x_updated, xd);
      }
      EXITING_REAL_TIME_CRITICAL_CODE
    }
  }

  return 0;
}
