/**
 * @file   FunctionApproximatorRBFN.cpp
 * @brief  FunctionApproximatorRBFN class source file.
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

#include "functionapproximators/FunctionApproximatorRBFN.hpp"
#include "functionapproximators/ModelParametersRBFN.hpp"
#include "functionapproximators/MetaParametersRBFN.hpp"
#include "functionapproximators/BasisFunction.hpp"

#include "eigen/eigen_file_io.hpp"

#include <iostream>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/LU>

#include <nlohmann/json.hpp>

using namespace std;
using namespace Eigen;

namespace DmpBbo {

FunctionApproximatorRBFN::FunctionApproximatorRBFN(const MetaParametersRBFN *const meta_parameters, const ModelParametersRBFN *const model_parameters) 
:
  FunctionApproximator(meta_parameters,model_parameters)
{
  if (model_parameters!=NULL)
    preallocateMemory(model_parameters->getNumberOfBasisFunctions());
}

FunctionApproximatorRBFN::FunctionApproximatorRBFN(const ModelParametersRBFN *const model_parameters) 
:
  FunctionApproximator(model_parameters)
{
  preallocateMemory(model_parameters->getNumberOfBasisFunctions());
}

void FunctionApproximatorRBFN::preallocateMemory(int n_basis_functions)
{
  weights_prealloc_ = VectorXd(n_basis_functions);
  activations_one_prealloc_ = MatrixXd(1,n_basis_functions);
  activations_prealloc_ = MatrixXd(1,n_basis_functions);
}


FunctionApproximator* FunctionApproximatorRBFN::clone(void) const {
  // All error checking and cloning is left to the FunctionApproximator constructor.
  return new FunctionApproximatorRBFN(
    dynamic_cast<const MetaParametersRBFN*>(getMetaParameters()),
    dynamic_cast<const ModelParametersRBFN*>(getModelParameters())
    );
};

void FunctionApproximatorRBFN::predict(const Eigen::Ref<const Eigen::MatrixXd>& inputs, MatrixXd& outputs)
{
  if (!isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorLWPR::predict if you have not trained yet. Doing nothing." << endl;
    return;
  }

  const ModelParametersRBFN* model_parameters_rbfn = static_cast<const ModelParametersRBFN*>(getModelParameters());
  
  model_parameters_rbfn->weights(weights_prealloc_);
  
  int n_basis_functions = model_parameters_rbfn->getNumberOfBasisFunctions();
  
  bool only_one_sample = (inputs.rows()==1);
  if (only_one_sample)
  {
    ENTERING_REAL_TIME_CRITICAL_CODE
    
    // Get the basis function activations  
    model_parameters_rbfn->kernelActivations(inputs,activations_one_prealloc_);
      
    // Weight the basis function activations  
    for (int b=0; b<n_basis_functions; b++)
      activations_one_prealloc_.col(b).array() *= weights_prealloc_(b);
  
    // Sum over weighed basis functions
    outputs = activations_one_prealloc_.rowwise().sum();
    
    EXITING_REAL_TIME_CRITICAL_CODE
  }
  else 
  {
    int n_time_steps = inputs.rows();

    // The next two lines may not be real-time, as they may allocate memory.
    // (if the size are already correct, it will be realtime)
    activations_prealloc_.resize(n_time_steps,n_basis_functions);
    outputs.resize(n_time_steps,getExpectedOutputDim());
    
    // Get the basis function activations  
    model_parameters_rbfn->kernelActivations(inputs,activations_prealloc_);
      
    // Weight the basis function activations  
    for (int b=0; b<n_basis_functions; b++)
      activations_prealloc_.col(b).array() *= weights_prealloc_(b);
  
    // Sum over weighed basis functions
    outputs = activations_prealloc_.rowwise().sum();
  }
    
}

FunctionApproximatorRBFN* FunctionApproximatorRBFN::from_jsonpickle(nlohmann::json json) {
  MetaParametersRBFN* meta = NULL;
  if (json.contains("_meta_params"))
    meta = MetaParametersRBFN::from_jsonpickle(json["_meta_params"]);
  
  ModelParametersRBFN* model = NULL;
  if (json.contains("_model_params"))
    model = ModelParametersRBFN::from_jsonpickle(json["_model_params"]);
  
  return new FunctionApproximatorRBFN(meta,model);
}

}
