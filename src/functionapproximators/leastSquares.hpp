/**
 * @file   leastSquares.hpp
 * @brief  Header file for various least squares functions.
 * @author Freek Stulp
 *
 * This file is part of DmpBbo, a set of libraries and programs for the 
 * black-box optimization of dynamical movement primitives.
 * Copyright (C) 2018 Freek Stulp
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

#ifndef _LEAST_SQUARES_H_
#define _LEAST_SQUARES_H_

/** @defgroup LeastSquares Least Squares
 *  @ingroup FunctionApproximators
 */

#include <eigen3/Eigen/Core>

namespace DmpBbo {
  
 
/** (Regularized) weighted least squares with bias 
 * \param[in] inputs Input values. Size n_samples X n_input_dims
 * \param[in] targets Target values. Size n_samples X n_ouput_dims
 * \param[in] weights Weights, one for each sample. Size n_samples X 1
 * \param[in] use_offset Use linear model "y = a*x + offset" instead of "y = a*x". Default: true.
 * \param[in] regularization Regularization term for regularized least squares. Default: 0.0.
 * \param[in] min_weight Minimum weight taken into account for least squares. Samples with a lower weight are not included in the least squares regression. May lead to significant speed-up. See documentation in cpp file for more details. Default: 0.0.
 * \return Parameters of the linear model
 */
Eigen::MatrixXd weightedLeastSquares(
  const Eigen::Ref<const Eigen::MatrixXd>& inputs, 
  const Eigen::Ref<const Eigen::MatrixXd>& targets,
  const Eigen::Ref<const Eigen::VectorXd>& weights,
  bool use_offset=true,
  double regularization=0.0,
  double min_weight=0.0
  );

/** (Regularized) least squares with bias 
 * \param[in] inputs Input values. Size n_samples X n_input_dims
 * \param[in] targets Target values. Size n_samples X n_ouput_dims
 * \param[in] use_offset Use linear model "y = a*x + offset" instead of "y = a*x". Default: true.
 * \param[in] regularization Regularization term for regularized least squares. Default: 0.0.
 * \return Parameters of the linear model
 */
Eigen::MatrixXd leastSquares(
  const Eigen::Ref<const Eigen::MatrixXd>& inputs, 
  const Eigen::Ref<const Eigen::MatrixXd>& targets,
  bool use_offset=true,
  double regularization=0.0
  );

/** (Regularized) least squares with bias 
 * \param[in] inputs Input values. Size n_samples X n_input_dims
 * \param[in] beta Parameters of the linear model, i.e. y = beta[0]*x + beta[1]
 * \param[out] outputs Predicted output values. Size n_samples X n_ouput_dims
 */
void linearPrediction(
  const Eigen::Ref<const Eigen::MatrixXd>& inputs, 
  const Eigen::Ref<const Eigen::VectorXd>& beta,
  Eigen::MatrixXd& outputs
);

}

#endif // _LEAST_SQUARES_H_


