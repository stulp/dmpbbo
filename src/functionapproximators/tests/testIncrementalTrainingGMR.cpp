/**
 * \file testIncrementalTrainingGMR.cpp
 * \author Gennaro Raiola
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

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <ctime>
#include <boost/filesystem.hpp>

#include "functionapproximators/FunctionApproximatorGMR.hpp"
#include "functionapproximators/MetaParametersGMR.hpp"
#include "functionapproximators/ModelParametersGMR.hpp"

#include "getFunctionApproximatorsVector.hpp"
#include "../demos/targetFunction.hpp"

using namespace DmpBbo;
using namespace Eigen;
using namespace std;

int main(int n_args, char** args)
{
  // Gaussian Mixture Regression (GMR)
  int dim_in = 1;
  int number_of_gaussians = 5;
  MetaParametersGMR* meta_parameters_gmr = new MetaParametersGMR(dim_in,number_of_gaussians);
  FunctionApproximatorGMR* fa_ptr = new FunctionApproximatorGMR(meta_parameters_gmr);

  // Re-train the model 3 times with the same data.
  // For a real case, the training data should be different at each iteration.
  // Check if the error gets smaller.
  for (int i = 0; i<3; i++)
  {
      MatrixXd inputs, targets, outputs;
      VectorXi n_samples_per_dim = VectorXi::Constant(1,30);
      targetFunction(n_samples_per_dim,inputs,targets);

      fa_ptr->trainIncremental(inputs,targets);

      // Do predictions and compute mean absolute error
      outputs.resize(inputs.rows(),fa_ptr->getExpectedOutputDim());
      fa_ptr->predict(inputs,outputs);

      MatrixXd abs_error = (targets.array()-outputs.array()).abs();
      VectorXd mean_abs_error_per_output_dim = abs_error.colwise().mean();

      cout << fixed << setprecision(5);
      cout << "         Mean absolute error ";
      if (mean_abs_error_per_output_dim.size()>1) cout << " (per dimension)";
      cout << ": " << mean_abs_error_per_output_dim.transpose();
      cout << "   \t(range of target data is " << targets.colwise().maxCoeff().array()-targets.colwise().minCoeff().array() << ")\n";

  }

  delete meta_parameters_gmr;
  delete fa_ptr;

  return 0;

}

