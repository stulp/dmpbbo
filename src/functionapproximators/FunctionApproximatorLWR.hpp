/**
 * @file   FunctionApproximatorLWR.hpp
 * @brief  FunctionApproximatorLWR class header file.
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

#ifndef _FUNCTION_APPROXIMATOR_LWR_H_
#define _FUNCTION_APPROXIMATOR_LWR_H_

#include "functionapproximators/FunctionApproximator.hpp"

#include <nlohmann/json_fwd.hpp>

/** @defgroup LWR Locally Weighted Regression (LWR)
 *  @ingroup FunctionApproximators
 */

namespace DmpBbo {
  
// Forward declarations
class ModelParametersLWR;

/** \brief LWR (Locally Weighted Regression) function approximator
 * \ingroup FunctionApproximators
 * \ingroup LWR  
 */
class FunctionApproximatorLWR : public FunctionApproximator
{
public:

  /** Initialize a function approximator with model parameters
   *  \param[in] model_parameters The parameters of the (previously) trained model.
   */
  FunctionApproximatorLWR(ModelParametersLWR* model_parameters);

  FunctionApproximatorLWR(int expected_input_dim, const Eigen::VectorXi& n_basis_functions_per_dim, double intersection_height=0.5, double regularization=0.0, bool asymmetric_kernels=false);
  
  ~FunctionApproximatorLWR(void);
  
  static FunctionApproximatorLWR* from_jsonpickle(nlohmann::json json);
  // https://github.com/nlohmann/json/issues/1324
  friend void to_json(nlohmann::json& j, const FunctionApproximatorLWR& m);
  //friend void from_json(const nlohmann::json& j, FunctionApproximatorLWR& m);
  
  /** Query the function approximator to make a prediction
   *  \param[in]  inputs   Input values of the query
   *  \param[out] outputs  Predicted output values
   *
   * \remark This method should be const. But third party functions which is called in this function
   * have not always been implemented as const (Examples: LWPRObject::predict).
   * Therefore, this function cannot be const.
   *
   * This function is realtime if inputs.rows()==1 (i.e. only one input sample is provided), and the
   * memory for outputs is preallocated.
   */
	void predict(const Eigen::Ref<const Eigen::MatrixXd>& inputs, Eigen::MatrixXd& outputs) const;

	std::string toString(void) const;
	
private:  
  /** The model parameters of the function approximator.
   */
  ModelParametersLWR* model_parameters_;
  
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  FunctionApproximatorLWR(void) {};
   
  void preallocateMemory(int n_basis_functions);
  
  /** Preallocated memory for one time step, required to make the predict() function real-time. */
  mutable Eigen::MatrixXd lines_one_prealloc_;
  
  /** Preallocated memory for one time step, required to make the predict() function real-time. */
  mutable Eigen::MatrixXd activations_one_prealloc_;
  
  /** Preallocated memory to make things more efficient. */
  mutable Eigen::MatrixXd lines_prealloc_;
  
  /** Preallocated memory to make things more efficient. */
  mutable Eigen::MatrixXd activations_prealloc_;
  
};

}

#endif // _FUNCTION_APPROXIMATOR_LWR_H_


