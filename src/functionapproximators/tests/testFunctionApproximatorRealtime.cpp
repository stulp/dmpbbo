/**
 * \file testFunctionApproximatorRealtime.cpp
 * \author Freek Stulp
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
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "functionapproximators/FunctionApproximator.hpp"

//#include "getFunctionApproximatorsVector.hpp"
#include "testTargetFunction.hpp"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

int main(int n_args, char** args)
{
  for (int n_input_dims = 1; n_input_dims <= 2; n_input_dims++) {
    vector<FunctionApproximator*> fas;
    getFunctionApproximatorsVector(n_input_dims, fas);

    for (unsigned int i_fa = 0; i_fa < fas.size(); i_fa++) {
      FunctionApproximator* cur_fa = fas[i_fa];

      VectorXi n_samples_per_dim = VectorXi::Constant(1, 30);
      if (n_input_dims == 2) n_samples_per_dim = VectorXi::Constant(2, 10);

      MatrixXd inputs, targets, outputs;
      targetFunction(n_samples_per_dim, inputs, targets);

      // Here, we time the predict function on single inputs, i.e. typical usage
      // in a real-time loop on a robot. We check if memory is allocated with
      // ENTERING_REAL_TIME_CRITICAL_CODE

      MatrixXd input = inputs.row(0);
      MatrixXd output(1, outputs.cols());
      for (int ii = 0; ii < inputs.rows(); ii++) {
        input = inputs.row(ii);
        ENTERING_REAL_TIME_CRITICAL_CODE
        cur_fa->predict(input, output);
        EXITING_REAL_TIME_CRITICAL_CODE
      }

      delete cur_fa;
    }
  }

  return 0;
}
