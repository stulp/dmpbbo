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
#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/BasisFunction.hpp"

#include "eigen/eigen_file_io.hpp"

#include <iostream>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/LU>

#include <nlohmann/json.hpp>


using namespace std;
using namespace Eigen;

namespace DmpBbo {

FunctionApproximatorLWR::FunctionApproximatorLWR(const MetaParametersLWR *const meta_parameters, const ModelParametersLWR *const model_parameters) 
:
  FunctionApproximator(meta_parameters,model_parameters)
{
  if (model_parameters!=NULL)
    preallocateMemory(model_parameters->getNumberOfBasisFunctions());
}

FunctionApproximatorLWR::FunctionApproximatorLWR(const ModelParametersLWR *const model_parameters) 
:
  FunctionApproximator(model_parameters)
{
  preallocateMemory(model_parameters->getNumberOfBasisFunctions());
}

void FunctionApproximatorLWR::preallocateMemory(int n_basis_functions)
{
  lines_one_prealloc_ = MatrixXd(1,n_basis_functions);
  activations_one_prealloc_ = MatrixXd(1,n_basis_functions);
  
  lines_prealloc_ = MatrixXd(1,n_basis_functions);
  activations_prealloc_ = MatrixXd(1,n_basis_functions);
}

FunctionApproximatorLWR::FunctionApproximatorLWR(int expected_input_dim, const Eigen::VectorXi& n_bfs_per_dim, double intersection_height, double regularization, bool asymmetric_kernels)
:
  FunctionApproximator(new MetaParametersLWR(expected_input_dim,n_bfs_per_dim,intersection_height,regularization,asymmetric_kernels))
{  
  
}

FunctionApproximator* FunctionApproximatorLWR::clone(void) const {
  // All error checking and cloning is left to the FunctionApproximator constructor.
  return new FunctionApproximatorLWR(
    dynamic_cast<const MetaParametersLWR*>(getMetaParameters()),
    dynamic_cast<const ModelParametersLWR*>(getModelParameters())
    );
};

void FunctionApproximatorLWR::predict(const Eigen::Ref<const Eigen::MatrixXd>& inputs, MatrixXd& outputs)
{

  if (!isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorLWPR::predict if you have not trained yet. Doing nothing." << endl;
    return;
  }
  
  // The following line of code took a long time to decide on.
  // The member FunctionApproximator::model_parameters_ (which we access through
  // getModelParameters()) is of class ModelParameters, not ModelParametersLWR.
  // So within this function, we need to cast it to ModelParametersLWR in order to make predictions.
  // There are three options to do this:
  //
  // 1) use a dynamic_cast. This is really the best way to do it, but the execution of dynamic_cast
  //    can take relatively long, so we should avoid calling it in this time-critical function
  //    predict() function. (note: because it doesn't matter so much for the non time-critical
  //    train() function above, we  use the preferred dynamic_cast<MetaParametersLWR*> as we should)
  //
  // 2) move the model_parameters_ member from FunctionApproximator to FunctionApproximatorLWR, and 
  //    make it ModelParametersLWR instead of ModelParameters. This, however, will lead to lots of 
  //    code duplication, because each derived function approximator class will have to do this.
  //
  // 3) Do a static_cast. The static cast does not do checking like dynamic_cast, so we have to be
  //    really sure that getModelParameters returns a ModelParametersLWR. The only way in which this 
  //    could wrong is if someone calls setModelParameters() with a different derived class. And
  //    this is near-impossible, because setModelParameters is protected within 
  //    FunctionApproximator, and a derived class would be really dumb to set ModelParametersAAA 
  //    with setModelParameters and expect getModelParameters to return ModelParametersBBB. 
  //
  // So I decided to go with 3) because it is fast and does not lead to code duplication, 
  // and only real dumb derived classes can cause trouble ;-)
  //
  // Note: The execution time difference between 2) and 3) is negligible:  
  //   No cast    : 8.90 microseconds/prediction of 1 input sample
  //   Static cast: 8.91 microseconds/prediction of 1 input sample
  //
  // There, ~30 lines of comment for one line of code ;-) 
  //                                            (mostly for me to remember why it is like this) 
  const ModelParametersLWR* model_parameters_lwr = static_cast<const ModelParametersLWR*>(getModelParameters());
  
  bool only_one_sample = (inputs.rows()==1);
  if (only_one_sample)
  {
    ENTERING_REAL_TIME_CRITICAL_CODE

    // Only 1 sample, so real-time execution is possible. No need to allocate memory.
    model_parameters_lwr->getLines(inputs, lines_one_prealloc_);

    // Weight the values for each line with the normalized basis function activations  
    model_parameters_lwr->kernelActivations(inputs,activations_one_prealloc_);
  
    outputs = (lines_one_prealloc_.array()*activations_one_prealloc_.array()).rowwise().sum();  
    
    EXITING_REAL_TIME_CRITICAL_CODE
    
  }
  else
  {
    
    int n_time_steps = inputs.rows();
    int n_basis_functions = model_parameters_lwr->getNumberOfBasisFunctions();
    
    // The next two lines are not be real-time, as they allocate memory
    lines_prealloc_.resize(n_time_steps,n_basis_functions);
    activations_prealloc_.resize(n_time_steps,n_basis_functions);
    outputs.resize(n_time_steps,getExpectedOutputDim());
    
    model_parameters_lwr->getLines(inputs, lines_prealloc_);

    // Weight the values for each line with the normalized basis function activations  
    model_parameters_lwr->kernelActivations(inputs,activations_prealloc_);
  
    outputs = (lines_prealloc_.array()*activations_prealloc_.array()).rowwise().sum();  
    
  }
  
}

FunctionApproximatorLWR* FunctionApproximatorLWR::from_jsonpickle(nlohmann::json json) {

  MetaParametersLWR* meta = NULL;
  if (json.contains("_meta_params")) {
    meta = MetaParametersLWR::from_jsonpickle(json["_meta_params"]);
  }
  
  ModelParametersLWR* model = NULL;
  if (json.contains("_model_params")) {
    model = ModelParametersLWR::from_jsonpickle(json["_model_params"]);
  }
  
  return new FunctionApproximatorLWR(meta,model);
}

}
