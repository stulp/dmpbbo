/**
 * @file   eigen_json.hpp
 * @author Freek Stulp
 *
 * This file is part of DmpBbo, a set of libraries and programs for the
 * black-box optimization of dynamical movement primitives.
 * Copyright (C) 2022 Freek Stulp
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

#ifndef _EIGEN_JSON_HPP_
#define _EIGEN_JSON_HPP_

#include <eigen3/Eigen/Core>
#include <iostream>
#include <nlohmann/json.hpp>

namespace Eigen {

/** Output a matrix as json.
 * \param[out] j The json output
 * \param[in] matrix The matrix
 */
template <typename Derived>
void to_json(nlohmann::json& j, const MatrixBase<Derived>& matrix);

/** Parse a matrix from json.
 * \param[in] j The json input
 * \param[out] matrix The parsed matrix
 */
template <typename Derived>
void from_json(const nlohmann::json& j, MatrixBase<Derived>& matrix);

template <typename Derived>
void to_json(nlohmann::json& j, const MatrixBase<Derived>& matrix)
{
  for (int row = 0; row < matrix.rows(); ++row) {
    nlohmann::json column = nlohmann::json::array();
    for (int col = 0; col < matrix.cols(); ++col) {
      column.push_back(matrix(row, col));
    }
    j.push_back(column);
  }
}

#include "eigen_json.tpp"


}  // namespace Eigen

#endif  //  #ifndef _EIGEN_JSON_HPP_