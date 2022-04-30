/**
 * @file   FunctionApproximator.hpp
 * @brief  FunctionApproximator class header file.
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

/** \defgroup FunctionApproximators Function Approximators Module
 */

#ifndef _FUNCTIONAPPROXIMATOR_H_
#define _FUNCTIONAPPROXIMATOR_H_

#define EIGEN_RUNTIME_NO_MALLOC  // Enable runtime tests for allocations

#include <eigen3/Eigen/Core>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

namespace DmpBbo {

/** \brief Pure abstract class for all function approximators.
 *  \ingroup FunctionApproximators
 */
class FunctionApproximator {
 public:
  /** Initialize a function approximator. */
  FunctionApproximator(){};

  virtual ~FunctionApproximator(void){};

  /** Query the function approximator to make a prediction
   *  \param[in]  inputs   Input values of the query (n_samples X n_input_dims)
   *  \param[out] outputs  Predicted output values (n_samples X n_output_dims)
   *
   * This function does one prediction for each row in inputs. This function
   * is not real-time, due to memory allocation.
   */
  virtual void predict(const Eigen::Ref<const Eigen::MatrixXd>& inputs,
                       Eigen::MatrixXd& outputs) const = 0;

  /** Query the function approximator to make a prediction.
   *
   *  \param[in]  input   Input value of the query (1 x n_input_dims)
   *  \param[out] output  Predicted output values (n_output_dims x 1)
   *
   * This function is real-time; there will be no memory allocation. In
   * constrast to predict(), this function make a prediction for one input only.
   */
  virtual void predictRealTime(
      const Eigen::Ref<const Eigen::RowVectorXd>& input,
      Eigen::VectorXd& output) const = 0;

  /** Print to output stream.
   *
   *  \param[in] output  Output stream to which to write to
   *  \param[in] function_approximator Function approximator to write
   *  \return    Output stream
   */
  friend std::ostream& operator<<(
      std::ostream& output, const FunctionApproximator& function_approximator);

  /** Read an object from json.
   *  \param[in]  j   json input
   *  \param[out] obj The object read from json
   *
   * See also: https://github.com/nlohmann/json/issues/1324
   */
  friend void from_json(const nlohmann::json& j, FunctionApproximator*& obj);

  /** Write an object to json.
   *  \param[in] obj The object to write to json
   *  \param[out]  j json output
   *
   * See also:
   *   https://github.com/nlohmann/json/issues/1324
   *   https://github.com/nlohmann/json/issues/716
   */
  inline friend void to_json(nlohmann::json& j,
                             const FunctionApproximator* const& obj)
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
  virtual void to_json_helper(nlohmann::json& j) const = 0;
};

}  // namespace DmpBbo

#endif  // _FUNCTIONAPPROXIMATOR_H_

namespace DmpBbo {

/** \page page_func_approx Function Approximators

This page provides an  overview of the implementation of function approximators
in the \c functionapproximators/ module.

It is assumed you have read about the theory behind function approximators in
the tutorial <a
href="https://github.com/stulp/dmpbbo/blob/master/tutorial/functionapproximators.md">tutorial/functionapproximators.md</a>.

Since v2.* of dmpbbo, the idea is to train a function approximator in Python,
write the trained result to json with jsonpickle, and read the json into C++ for
real-time prediction.

There are two functions fo prediction:

\li FunctionApproximator::predictRealTime(), which takes one input of size 1 x
n_input_dims as an input, and whose output is a prediction of 1 x n_output_dims
(n_output_dims is usually 1 in the context of dmpbbo)

\li FunctionApproximator::predict(), this takes multiple samples at once (in a
n_samples x n_input_dims matrix), and provides one prediction for each sample
(in a n_samples x n_output_dims matrix). This function is not real-time. It is
used in for instance Dmp::analyticalSolution().

In version 1.* of dmpbbo, the training of function approximators was also
possible in C++. To reduce duplicated code in version 2.*, all training is done
in Python. The C++ code is only there to enable real-time prediction (i.e.
without memory allocations).

 */

}
