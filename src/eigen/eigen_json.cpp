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

#include <iostream>

#include "eigen/eigen_json.hpp"

using namespace std;
using namespace Eigen;

namespace DmpBbo {
  
void from_json(const nlohmann::json& j, VectorXi& vector)
{
    //using Scalar = typename MatrixXi::Scalar;
    using Index = typename MatrixXi::Index;

    if (j.is_array())
    {
        vector.resize(j.size());
        for (std::size_t ii = 0; ii < j.size(); ++ii)
        {
            const auto& value = j.at(ii);
            vector(static_cast<Index>(ii)) = value.get<int>();
        }
    }
    else
    {
        vector.resize(1);
        vector(0) = j.get<int>();
    }
}
  
void from_json(const nlohmann::json& j, VectorXd& vector)
{
    using Scalar = typename MatrixXd::Scalar;
    using Index = typename MatrixXd::Index;

    if (j.is_array())
    {
        vector.resize(j.size());
        for (std::size_t ii = 0; ii < j.size(); ++ii)
        {
            const auto& value = j.at(ii);
            vector(static_cast<Index>(ii)) = value.get<Scalar>();
        }
    }
    else
    {
        vector.resize(1);
        vector(0) = j.get<Scalar>();
    }
}
  
void from_json(const nlohmann::json& j, MatrixXd& matrix)
{
    using Scalar = typename MatrixXd::Scalar;
    using Index = typename MatrixXd::Index;

    bool resized = false;
    for (std::size_t row = 0; row < j.size(); ++row)
    {
        const auto& jrow = j.at(row);
        if (jrow.is_array())
        {
            if (!resized) 
            {
              matrix.resize(j.size(),jrow.size());
              resized = true;
            }
            for (std::size_t col = 0; col < jrow.size(); ++col)
            {
                const auto& value = jrow.at(col);
                matrix(static_cast<Index>(row), static_cast<Index>(col)) = value.get<Scalar>();
            }
        }
        else
        {
            if (!resized) 
            {
              matrix.resize(j.size(),1);
              resized = true;
            }
            matrix(static_cast<Index>(row), 0) = jrow.get<Scalar>();
        }
    }
}

int from_json_to_double(const nlohmann::json& j) {
  
  // value
  if (j.contains("value")) {
    return j.at("value");
  }
  
  // values
  if (j.contains("values")) {
    MatrixXd matrix;
    from_json(j.at("values"), matrix); 
    
    if (matrix.rows()!=1) {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "'values' should have 1 row, but has " << matrix.rows() << endl;
    }
    if (matrix.cols()!=1) {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "'values' should have 1 cols, but has " << matrix.rows() << endl;
    }
    
    return matrix(0,0);
  }

  // double
  double d = j;
  return d;
  
}


}
