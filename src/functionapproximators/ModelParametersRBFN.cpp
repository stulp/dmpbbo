/**
 * @file   ModelParametersRBFN.cpp
 * @brief  ModelParametersRBFN class source file.
 * @author Freek Stulp
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

#include "functionapproximators/ModelParametersRBFN.hpp"
#include "functionapproximators/BasisFunction.hpp"


#include "eigen/eigen_json.hpp"

#include <iostream>
#include <fstream>
#include <vector>

#include <eigen3/Eigen/Core>



using namespace std;
using namespace Eigen;
using namespace nlohmann;

namespace DmpBbo {

ModelParametersRBFN::ModelParametersRBFN(const Eigen::MatrixXd& centers, const Eigen::MatrixXd& widths, const Eigen::MatrixXd& weights) 
:
  centers_(centers),
  widths_(widths),
  weights_(weights)
{
#ifndef NDEBUG // Variables below are only required for asserts; check for NDEBUG to avoid warnings.
  int n_basis_functions = centers.rows();
  int n_dims = centers.cols();
#endif  
  assert(n_basis_functions==widths_.rows());
  assert(n_dims           ==widths_.cols());
  assert(n_basis_functions==weights_.rows());
  assert(1                ==weights_.cols());
};

void ModelParametersRBFN::kernelActivations(const Eigen::Ref<const Eigen::MatrixXd>& inputs, Eigen::MatrixXd& kernel_activations) const
{
  ENTERING_REAL_TIME_CRITICAL_CODE
  bool normalized_basis_functions=false;  
  bool asymmetric_kernels=false;
  BasisFunction::Gaussian::activations(centers_,widths_,inputs,kernel_activations,
    normalized_basis_functions,asymmetric_kernels);
  EXITING_REAL_TIME_CRITICAL_CODE
}

ModelParametersRBFN* ModelParametersRBFN::from_jsonpickle(const nlohmann::json& json) {
  
  MatrixXd centers = json.at("centers").at("values");
  MatrixXd widths = json.at("widths").at("values");
  MatrixXd weights = json.at("weights").at("values");
  
  return new ModelParametersRBFN(centers,widths,weights);
}

void to_json(nlohmann::json& j, const ModelParametersRBFN& obj) {
  
  j["centers_"] = obj.centers_;
  j["widths_"] = obj.widths_;
  j["weights_"] = obj.weights_;
  
  // for jsonpickle
  j["py/object"] = "dynamicalsystems.ModelParametersRBFN.ModelParametersRBFN";
}

string ModelParametersRBFN::toString(void) const
{
  nlohmann::json j;
  to_json(j,*this);
  return j.dump(4);
}

}


