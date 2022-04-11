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

// Below code was adapted from
// https://gitlab.com/Simox/simox/-/blob/master/SimoxUtility/json/eigen_conversion.h

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
void from_json(const nlohmann::json& j, MatrixBase<Derived>& vector);

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

template <typename Derived>
void from_json(const nlohmann::json& j, Eigen::MatrixBase<Derived>& matrix)
{
  using Scalar = typename Eigen::MatrixBase<Derived>::Scalar;
  using Index = typename Eigen::MatrixBase<Derived>::Index;

  bool resized = false;
  for (std::size_t row = 0; row < j.size(); ++row) {
    const auto& jrow = j.at(row);
    if (jrow.is_array()) {
      if (!resized) {
        matrix.derived().resize(j.size(), jrow.size());
        resized = true;
      }

      for (std::size_t col = 0; col < jrow.size(); ++col) {
        const auto& value = jrow.at(col);
        matrix(static_cast<Index>(row), static_cast<Index>(col)) =
            value.get<Scalar>();
      }
    } else {
      if (!resized) {
        // https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html#title4
        matrix.derived().resize(j.size(), 1);
        resized = true;
      }
      matrix(static_cast<Index>(row), 0) = jrow.get<Scalar>();
    }
  }
}

/**
 * Read int from json. can be "4.0", "[4.0]", "[[4.0]]"
 */
inline double from_json_to_double(const nlohmann::json& j)
{
  if (j.contains("value")) {
    double dv = j.at("value");
    return dv;
  }

  // values
  if (j.contains("values")) {
    Eigen::MatrixXd matrix;
    from_json(j.at("values"), matrix);

    if (matrix.rows() != 1) {
      std::cerr << __FILE__ << ":" << __LINE__ << ":";
      std::cerr << "'values' should have 1 row, but has " << matrix.rows()
                << std::endl;
    }
    if (matrix.cols() != 1) {
      std::cerr << __FILE__ << ":" << __LINE__ << ":";
      std::cerr << "'values' should have 1 cols, but has " << matrix.rows()
                << std::endl;
    }

    return matrix(0, 0);
  }

  // double
  double d = j;
  return d;
}

}  // namespace Eigen

#endif  //  #ifndef _EIGEN_JSON_HPP_