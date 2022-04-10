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

  /** Constructor for the model parameters of the LWPR function approximator.
   *  \param[in] centers Centers of the basis functions
   *  \param[in] widths  Widths of the basis functions. 
   *  \param[in] slopes  Slopes of the line segments. 
   *  \param[in] offsets Offsets of the line segments, i.e. the value of the line segment at its intersection with the y-axis.
   * \param[in] asymmetric_kernels Whether to use asymmetric kernels or not, cf MetaParametersLWR::asymmetric_kernels()
   * \param[in] lines_pivot_at_max_activation Whether line models should pivot at x=0 (false), or at the center of the kernel (x=x_c)
   */
  FunctionApproximatorLWR(const Eigen::MatrixXd& centers, const Eigen::MatrixXd& widths, const Eigen::MatrixXd& slopes, const Eigen::MatrixXd& offsets, bool asymmetric_kernels=false, bool lines_pivot_at_max_activation=false);
  
  ~FunctionApproximatorLWR(void) {};
  
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

  /** Set whether the offsets should be adapted so that the line segments pivot around the mode of
   * the basis function, rather than the intersection with the y-axis.
   * \param[in] lines_pivot_at_max_activation Whether to pivot around the mode or not.
   *
   */
  void set_lines_pivot_at_max_activation(bool lines_pivot_at_max_activation);

  /** Whether to return slopes as angles or slopes in ModelParametersLWR::getParameterVectorAll()
   * \param[in] slopes_as_angles Whether to return as slopes (true) or angles (false)
   * \todo Implement and document
   */
  void set_slopes_as_angles(bool slopes_as_angles);
  
	std::string toString(void) const;
	
private:  
  /** The model parameters of the function approximator.
   */
  int n_basis_functions_;
  Eigen::MatrixXd centers_; // n_centers X n_dims
  Eigen::MatrixXd widths_;  // n_centers X n_dims
  Eigen::MatrixXd slopes_;  // n_centers X n_dims
  Eigen::VectorXd offsets_; // n_centers X 1

  bool asymmetric_kernels_;
  bool lines_pivot_at_max_activation_;
  bool slopes_as_angles_;
  
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  FunctionApproximatorLWR(void) {};
   
  /** Preallocated memory for one time step, required to make the predict() function real-time. */
  mutable Eigen::MatrixXd lines_one_prealloc_;
  
  /** Preallocated memory for one time step, required to make the predict() function real-time. */
  mutable Eigen::MatrixXd activations_one_prealloc_;
  
  /** Preallocated memory to make things more efficient. */
  mutable Eigen::MatrixXd lines_prealloc_;
  
  /** Preallocated memory to make things more efficient. */
  mutable Eigen::MatrixXd activations_prealloc_;
  
  /** Get the output of each linear model (unweighted) for the given inputs.
   * \param[in] inputs The inputs for which to compute the output of the lines models (size: n_samples X  n_input_dims)
   * \param[out] lines The output of the linear models (size: n_samples X n_basis_functions) 
   *
   * If "lines" is passed as a Matrix of correct size (n_samples X n_basis_functions), this function
   * will not allocate any memory, and is real-time.
   */
  void getLines(const Eigen::Ref<const Eigen::MatrixXd>& inputs, Eigen::MatrixXd& lines) const;
};

}

#endif // _FUNCTION_APPROXIMATOR_LWR_H_


