/**
 * @file   LeastSquares.hpp
 * @brief  LeastSquares class header file.
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

#ifndef _LEAST_SQUARES_H_
#define _LEAST_SQUARES_H_

/** @defgroup Least Squares
 *  @ingroup FunctionApproximators
 */

#include <eigen3/Eigen/Core>

namespace DmpBbo {
  
 
/** (Regularized) weighted least squares with bias */
Eigen::MatrixXd weightedLeastSquares(
  const Eigen::Ref<const Eigen::MatrixXd>& inputs, 
  const Eigen::Ref<const Eigen::MatrixXd>& targets,
  const Eigen::Ref<const Eigen::VectorXd>& weights,
  bool use_offset=true,
  double regularization=0.0,
  double min_weight=0.0
  );

/** (Regularized) least squares with bias */
Eigen::MatrixXd leastSquares(
  const Eigen::Ref<const Eigen::MatrixXd>& inputs, 
  const Eigen::Ref<const Eigen::MatrixXd>& targets,
  bool use_offset=true,
  double regularization=0.0
  )
{
  Eigen::VectorXd weights = Eigen::VectorXd::Ones(inputs.rows());
  return weightedLeastSquares(inputs,targets,weights,use_offset,regularization);
}

/** Linear prediction */
void linearPrediction(
  const Eigen::Ref<const Eigen::MatrixXd>& inputs, 
  const Eigen::Ref<const Eigen::VectorXd>& beta,
  Eigen::MatrixXd& outputs
);

}

#endif // _LEAST_SQUARES_H_


