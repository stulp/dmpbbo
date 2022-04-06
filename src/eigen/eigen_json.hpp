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

#pragma once

#include <eigen3/Eigen/Core>
#include <nlohmann/json.hpp>

namespace DmpBbo {
  
void from_json(const nlohmann::json& j, Eigen::VectorXd& vector);

void from_json(const nlohmann::json& j, Eigen::VectorXi& vector);

void from_json(const nlohmann::json& j, Eigen::MatrixXd& matrix);

//int from_json_to_int(const nlohmann::json& j);

/**
 * Read int from json. can be "4.0", "[4.0]", "[[4.0]]"
 */
int from_json_to_double(const nlohmann::json& j);


}