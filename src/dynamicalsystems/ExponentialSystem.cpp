/**
 * @file   ExponentialSystem.cpp
 * @brief  ExponentialSystem class source file.
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

#include "dynamicalsystems/ExponentialSystem.hpp"

#include <eigen3/Eigen/Core>
#include <nlohmann/json.hpp>

#include "eigenutils/eigen_json.hpp"
#include "eigenutils/eigen_realtime_check.hpp"

using namespace std;
using namespace Eigen;

namespace DmpBbo {

ExponentialSystem::ExponentialSystem(double tau, Eigen::VectorXd x_init,
                                     Eigen::VectorXd x_attr, double alpha)
    : DynamicalSystem(1, tau, x_init), x_attr_(x_attr), alpha_(alpha)
{
}

ExponentialSystem::~ExponentialSystem(void) {}

void ExponentialSystem::differentialEquation(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    Eigen::Ref<Eigen::VectorXd> xd) const
{
  ENTERING_REAL_TIME_CRITICAL_CODE
  xd.noalias() = alpha_ * (x_attr_ - x) / tau();
  EXITING_REAL_TIME_CRITICAL_CODE
}

void ExponentialSystem::analyticalSolution(const VectorXd& ts, MatrixXd& xs,
                                           MatrixXd& xds) const
{
  int n_time_steps = ts.size();

  // Usually, we expect xs and xds to be of size n_time_steps X dim(), so we
  // resize to that. However, if the input matrices were of size dim() X
  // n_time_steps, we return the matrices of that size by doing a
  // transposeInPlace at the end. That way, the user can also request dim() X
  // n_time_steps sized matrices.
  bool caller_expects_transposed =
      (xs.rows() == dim() && xs.cols() == n_time_steps);

  // Prepare output arguments to be of right size (Eigen does nothing if already
  // the right size)
  xs.resize(n_time_steps, dim());
  xds.resize(n_time_steps, dim());

  VectorXd val_range = x_init() - x_attr_;

  VectorXd exp_term = -alpha_ * ts / tau();
  exp_term = exp_term.array().exp().transpose();
  VectorXd pos_scale = exp_term;
  VectorXd vel_scale = -(alpha_ / tau()) * exp_term;

  xs = val_range.transpose().replicate(n_time_steps, 1).array() *
       pos_scale.replicate(1, dim()).array();
  xs += x_attr_.transpose().replicate(n_time_steps, 1);
  xds = val_range.transpose().replicate(n_time_steps, 1).array() *
        vel_scale.replicate(1, dim()).array();

  if (caller_expects_transposed) {
    xs.transposeInPlace();
    xds.transposeInPlace();
  }
}

void from_json(const nlohmann::json& j, ExponentialSystem*& obj)
{
  double tau = from_json_to_double(j.at("tau_"));
  double alpha = from_json_to_double(j.at("alpha_"));
  VectorXd y_init = j.at("y_init_").at("values");
  VectorXd y_attr = j.at("y_attr_").at("values");

  obj = new ExponentialSystem(tau, y_init, y_attr, alpha);
}

void ExponentialSystem::to_json_helper(nlohmann::json& j) const
{
  to_json_base(j);  // Get the json string from the base class
  j["alpha_"] = alpha_;
  j["y_attr_"] = x_attr_;

  string c("ExponentialSystem");
  j["py/object"] = "dynamicalsystems." + c + "." + c;  // for jsonpickle
}

}  // namespace DmpBbo
