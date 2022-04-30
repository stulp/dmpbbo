/**
 * @file   FunctionApproximatorRBFN.hpp
 * @brief  FunctionApproximatorRBFN class header file.
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

#ifndef _FUNCTION_APPROXIMATOR_RBFN_H_
#define _FUNCTION_APPROXIMATOR_RBFN_H_

#include <nlohmann/json_fwd.hpp>

#include "functionapproximators/FunctionApproximator.hpp"

/** @defgroup RBFN Radial Basis Function Network (RBFN)
 *  @ingroup FunctionApproximators
 */

namespace DmpBbo {

/** \brief RBFN (Radial Basis Function Network) function approximator
 * \ingroup FunctionApproximators
 * \ingroup RBFN
 */
class FunctionApproximatorRBFN : public FunctionApproximator {
 public:
  /** Constructor for the model parameters of the function approximator.
   *  \param[in] centers Centers of the basis functions
   *  \param[in] widths  Widths of the basis functions.
   *  \param[in] weights Weight of each basis function
   */
  FunctionApproximatorRBFN(const Eigen::MatrixXd& centers,
                           const Eigen::MatrixXd& widths,
                           const Eigen::MatrixXd& weights);

  /** Query the function approximator to make a prediction
   *  \param[in]  inputs   Input values of the query (n_samples X n_input_dims)
   *  \param[out] outputs  Predicted output values (n_samples X n_output_dims)
   *
   * This function does one prediction for each row in inputs. This function
   * is not real-time, due to memory allocation.
   */
  void predict(const Eigen::Ref<const Eigen::MatrixXd>& inputs,
               Eigen::MatrixXd& outputs) const;

  /** Query the function approximator to make a prediction.
   *
   *  \param[in]  input   Input value of the query (1 x n_input_dims)
   *  \param[out] output  Predicted output values (n_output_dims x 1)
   *
   * This function is real-time; there will be no memory allocation. In
   * constrast to predict(), this function make a prediction for one input only.
   */
  void predictRealTime(const Eigen::Ref<const Eigen::RowVectorXd>& input,
                       Eigen::VectorXd& output) const;

  /** Read an object from json.
   *  \param[in]  j   json input
   *  \param[out] obj The object read from json
   *
   * See also: https://github.com/nlohmann/json/issues/1324
   */
  friend void from_json(const nlohmann::json& j,
                        FunctionApproximatorRBFN*& obj);

  /** Write an object to json.
   *  \param[in] obj The object to write to json
   *  \param[out]  j json output
   *
   * See also:
   *   https://github.com/nlohmann/json/issues/1324
   *   https://github.com/nlohmann/json/issues/716
   */
  inline friend void to_json(nlohmann::json& j,
                             const FunctionApproximatorRBFN* const& obj)
  {
    obj->to_json_helper(j);
  }

 private:
  /** Write this object to json.
   *  \param[out]  j json output
   *
   * See also:
   *   https://github.com/nlohmann/json/issues/1324
   *   https://github.com/nlohmann/json/issues/716
   */
  void to_json_helper(nlohmann::json& j) const;

  int n_basis_functions_;
  Eigen::MatrixXd centers_;  // n_basis_functions_ X n_dims
  Eigen::MatrixXd widths_;   // n_basis_functions_ X n_dims
  Eigen::VectorXd weights_;  //                  1 X n_dims

  /** Preallocated memory for one time step, required to make the predict()
   * function real-time. */
  mutable Eigen::MatrixXd activations_one_prealloc_;
};

}  // namespace DmpBbo

#endif  // _FUNCTION_APPROXIMATOR_RBFN_H_
