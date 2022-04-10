/**
 * @file   FunctionApproximatorLWR.cpp
 * @brief  FunctionApproximatorLWR class source file.
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

#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/ModelParametersLWR.hpp"
#include "functionapproximators/BasisFunction.hpp"

#include "eigen/eigen_file_io.hpp"

#include <iostream>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/LU>

#include <nlohmann/json.hpp>


using namespace std;
using namespace Eigen;

namespace DmpBbo {

FunctionApproximatorLWR::FunctionApproximatorLWR(ModelParametersLWR* model_parameters) 
{
  model_parameters_ = model_parameters;
  preallocateMemory(model_parameters->getNumberOfBasisFunctions());
}

FunctionApproximatorLWR::~FunctionApproximatorLWR(void) 
{
  delete model_parameters_;
}



void FunctionApproximatorLWR::preallocateMemory(int n_basis_functions)
{
  lines_one_prealloc_ = MatrixXd(1,n_basis_functions);
  activations_one_prealloc_ = MatrixXd(1,n_basis_functions);
  
  lines_prealloc_ = MatrixXd(1,n_basis_functions);
  activations_prealloc_ = MatrixXd(1,n_basis_functions);
}

void FunctionApproximatorLWR::predict(const Eigen::Ref<const Eigen::MatrixXd>& inputs, MatrixXd& outputs)
{
  
  bool only_one_sample = (inputs.rows()==1);
  if (only_one_sample)
  {
    ENTERING_REAL_TIME_CRITICAL_CODE

    // Only 1 sample, so real-time execution is possible. No need to allocate memory.
    model_parameters_->getLines(inputs, lines_one_prealloc_);

    // Weight the values for each line with the normalized basis function activations  
    model_parameters_->kernelActivations(inputs,activations_one_prealloc_);
  
    outputs = (lines_one_prealloc_.array()*activations_one_prealloc_.array()).rowwise().sum();  
    
    EXITING_REAL_TIME_CRITICAL_CODE
    
  }
  else
  {
    
    int n_time_steps = inputs.rows();
    int n_basis_functions = model_parameters_->getNumberOfBasisFunctions();
    
    // The next two lines are not be real-time, as they allocate memory
    lines_prealloc_.resize(n_time_steps,n_basis_functions);
    activations_prealloc_.resize(n_time_steps,n_basis_functions);
    outputs.resize(n_time_steps,1);
    
    model_parameters_->getLines(inputs, lines_prealloc_);

    // Weight the values for each line with the normalized basis function activations  
    model_parameters_->kernelActivations(inputs,activations_prealloc_);
  
    outputs = (lines_prealloc_.array()*activations_prealloc_.array()).rowwise().sum();  
    
  }
  
}

FunctionApproximatorLWR* FunctionApproximatorLWR::from_jsonpickle(nlohmann::json json) {

  ModelParametersLWR* model = NULL;
  if (json.contains("_model_params")) {
    model = ModelParametersLWR::from_jsonpickle(json["_model_params"]);
  }
  
  return new FunctionApproximatorLWR(model);
}


string FunctionApproximatorLWR::toString(void) const
{
  std::stringstream s;
  s << "FunctionApproximatorLWR" << endl;
  return s.str();
}

}
