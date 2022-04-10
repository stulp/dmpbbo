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
#include <vector>
#include <iostream>
#include <eigen3/Eigen/Core>

#include "eigen/eigen_json.hpp"

#include <nlohmann/json.hpp>

using namespace std;
using namespace Eigen;

namespace DmpBbo {

TimeSystem::TimeSystem(double tau, bool count_down, std::string name)
: 
  DynamicalSystem(1, tau, VectorXd::Zero(1), VectorXd::Ones(1), name),
  count_down_(count_down)
{
  if (count_down_)
  {
    set_initial_state(VectorXd::Ones(1));
    set_attractor_state(VectorXd::Zero(1));
  }
}

TimeSystem::~TimeSystem(void)
{
}

DynamicalSystem* TimeSystem::clone(void) const
{
  return new TimeSystem(tau(),count_down(),name());
}

void TimeSystem::differentialEquation(
   const Eigen::Ref<const Eigen::VectorXd>& x, 
   Eigen::Ref<Eigen::VectorXd> xd) const
{
  ENTERING_REAL_TIME_CRITICAL_CODE
  
  // if state<1: xd = 1/obj.tau   (or for count_down=true, if state>0: xd = -1/obj.tau
  // else        xd = 0
  xd.resize(1);
  xd[0] = 0;
  
  if (count_down_)
  {
    if (x[0]>0)
      xd[0] = -1.0/tau();
  }
  else
  {
    if (x[0]<1)
      xd[0] = 1.0/tau();
  }
  
  EXITING_REAL_TIME_CRITICAL_CODE
}

void TimeSystem::analyticalSolution(const VectorXd& ts, MatrixXd& xs, MatrixXd& xds) const
{
  int T = ts.size();
  assert(T>0);

  // Usually, we expect xs and xds to be of size T X dim(), so we resize to that. However, if the
  // input matrices were of size dim() X T, we return the matrices of that size by doing a 
  // transposeInPlace at the end. That way, the user can also request dim() X T sized matrices.
  bool caller_expects_transposed = (xs.rows()==dim() && xs.cols()==T);

  // Prepare output arguments to be of right size (Eigen does nothing if already the right size)
  xs.resize(T,dim());
  xds.resize(T,dim());
  
  // Find first index at which the time is larger than tau. Then velocities should be set to zero.
  int velocity_stop_index = -1;
  int i=0;
  while (velocity_stop_index<0 && i<ts.size())
    if (ts[i++]>tau())
      velocity_stop_index = i-1;
    
  if (velocity_stop_index<0)
    velocity_stop_index = ts.size();

  if (count_down_)
  {
    xs.topRows(velocity_stop_index) = (-ts.segment(0,velocity_stop_index).array()/tau()).array()+1.0;
    xs.bottomRows(xs.size()-velocity_stop_index).fill(0.0);
  
    xds.topRows(velocity_stop_index).fill(-1.0/tau());
    xds.bottomRows(xds.size()-velocity_stop_index).fill(0.0);
  }
  else
  {
    xs.topRows(velocity_stop_index) = ts.segment(0,velocity_stop_index).array()/tau();
    xs.bottomRows(xs.size()-velocity_stop_index).fill(1.0);
  
    xds.topRows(velocity_stop_index).fill(1.0/tau());
    xds.bottomRows(xds.size()-velocity_stop_index).fill(0.0);
  }
  
  if (caller_expects_transposed)
  {
    xs.transposeInPlace();
    xds.transposeInPlace();
  }
}

TimeSystem* TimeSystem::from_jsonpickle(const nlohmann::json& json) {

  double tau = from_json_to_double(json.at("tau_"));
  int count_down_int = json.at("count_down_");
  bool count_down = count_down_int >0;
  string name = json.at("name_");

  return new TimeSystem(tau,count_down,name);
}

void to_json(nlohmann::json& j, const TimeSystem& obj) {
  obj.DynamicalSystem::to_json_base(j);
  
  j["count_down_"] = obj.count_down_;
  
  // for jsonpickle
  j["py/object"] = "dynamicalsystems.TimeSystem.TimeSystem";
}

string TimeSystem::toString(void) const
{
  nlohmann::json j;
  to_json(j,*this);
  return j.dump(4);
}

}
