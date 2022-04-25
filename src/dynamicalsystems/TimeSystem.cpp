/**
 * @file   TimeSystem.cpp
 * @brief  TimeSystem class source file.
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

#include "dynamicalsystems/TimeSystem.hpp"

#include <cmath>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

#include "eigenutils/eigen_json.hpp"
#include "eigenutils/eigen_realtime_check.hpp"

using namespace std;
using namespace Eigen;

namespace DmpBbo {

TimeSystem::TimeSystem(double tau, bool count_down)
    //                         Count-down goes from 1 to 0, so start at 1.0
    : DynamicalSystem(1, tau, (count_down ? 1.0 : 0.0) * VectorXd::Ones(1)),
      count_down_(count_down)
{
}

TimeSystem::~TimeSystem(void) {}

void TimeSystem::differentialEquation(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    Eigen::Ref<Eigen::VectorXd> xd) const
{
  ENTERING_REAL_TIME_CRITICAL_CODE

  // if state<1: xd = 1/obj.tau   (or for count_down=true, if state>0: xd =
  // -1/obj.tau else        xd = 0
  xd.resize(1);
  xd[0] = 0;

  if (count_down_) {
    if (x[0] > 0) xd[0] = -1.0 / tau();
  } else {
    if (x[0] < 1) xd[0] = 1.0 / tau();
  }

  EXITING_REAL_TIME_CRITICAL_CODE
}

void TimeSystem::analyticalSolution(const VectorXd& ts, MatrixXd& xs,
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

  // Find first index at which the time is larger than tau. Then velocities
  // should be set to zero.
  int velocity_stop_index = -1;
  int i = 0;
  while (velocity_stop_index < 0 && i < ts.size())
    if (ts[i++] > tau()) velocity_stop_index = i - 1;

  if (velocity_stop_index < 0) velocity_stop_index = ts.size();

  if (count_down_) {
    xs.topRows(velocity_stop_index) =
        (-ts.segment(0, velocity_stop_index).array() / tau()).array() + 1.0;
    xs.bottomRows(xs.size() - velocity_stop_index).fill(0.0);

    xds.topRows(velocity_stop_index).fill(-1.0 / tau());
    xds.bottomRows(xds.size() - velocity_stop_index).fill(0.0);
  } else {
    xs.topRows(velocity_stop_index) =
        ts.segment(0, velocity_stop_index).array() / tau();
    xs.bottomRows(xs.size() - velocity_stop_index).fill(1.0);

    xds.topRows(velocity_stop_index).fill(1.0 / tau());
    xds.bottomRows(xds.size() - velocity_stop_index).fill(0.0);
  }

  if (caller_expects_transposed) {
    xs.transposeInPlace();
    xds.transposeInPlace();
  }
}

void from_json(const nlohmann::json& j, TimeSystem*& obj)
{
  double tau = j.at("_tau");
  int count_down_int = j.at("_count_down");
  bool count_down = count_down_int > 0;

  obj = new TimeSystem(tau, count_down);
}

void TimeSystem::to_json_helper(nlohmann::json& j) const
{
  to_json_base(j);  // Get the json string from the base class
  j["_count_down"] = count_down_;
  j["class"] = "TimeSystem";
}

}  // namespace DmpBbo
