#pragma once

#include <eigen3/Eigen/Core>
#include <nlohmann/json.hpp>

namespace DmpBbo {
  
void from_json(const nlohmann::json& j, Eigen::VectorXd& vector);

void from_json(const nlohmann::json& j, Eigen::VectorXi& vector);

void from_json(const nlohmann::json& j, Eigen::MatrixXd& matrix);

}