/**
 * @file   DynamicalSystem.cpp
 * @brief  DynamicalSystem class source file.
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

#include "dynamicalsystems/DynamicalSystem.hpp"

#include <eigen3/Eigen/Core>
#include <nlohmann/json.hpp>

#include "dynamicalsystems/ExponentialSystem.hpp"
#include "dynamicalsystems/SigmoidSystem.hpp"
#include "dynamicalsystems/SpringDamperSystem.hpp"
#include "dynamicalsystems/TimeSystem.hpp"
#include "eigenutils/eigen_json.hpp"
#include "eigenutils/eigen_realtime_check.hpp"

using namespace std;
using namespace Eigen;

namespace DmpBbo {

DynamicalSystem::DynamicalSystem(int order, double tau, Eigen::VectorXd y_init)
    : dim_x_(y_init.size() * order), dim_y_(y_init.size()), tau_(tau)
{
  assert(order == 1 || order == 2);
  set_y_init(y_init);
  preallocateMemory();
}

DynamicalSystem::DynamicalSystem(double tau, Eigen::VectorXd y_init,
                                 int n_dims_x)
    : dim_x_(n_dims_x), dim_y_(y_init.size()), tau_(tau)
{
  set_y_init(y_init);
  preallocateMemory();
}

void DynamicalSystem::preallocateMemory()
{
  // Pre-allocate memory for Runge-Kutta integration
  k1_ = VectorXd(dim_x_);
  k2_ = VectorXd(dim_x_);
  k3_ = VectorXd(dim_x_);
  k4_ = VectorXd(dim_x_);

  input_k2_ = VectorXd(dim_x_);
  input_k3_ = VectorXd(dim_x_);
  input_k4_ = VectorXd(dim_x_);
}

DynamicalSystem::~DynamicalSystem(void) {}

void DynamicalSystem::get_y_init(Eigen::VectorXd& y_init) const
{
  if (dim_x_ == dim_y_)
    y_init = x_init_;
  else
    // x = [y z], return only y part
    y_init = x_init_.segment(0, dim_y_);
  // The upper statement would suffice. The if-then-else makes the semantics
  // clearer.
}

void DynamicalSystem::set_y_init(
    const Eigen::Ref<const Eigen::VectorXd>& y_init)
{
  assert(y_init.size() == dim_y_);
  if (dim_x_ == dim_y_) {
    x_init_ = y_init;
  } else {
    // All other cases: x_init_ = [y_init 0 0 ...]
    x_init_ = VectorXd::Zero(dim_x_);
    x_init_.segment(0, dim_y_) = y_init;
  }
}

void DynamicalSystem::integrateStart(const Eigen::VectorXd& y_init,
                                     Eigen::Ref<Eigen::VectorXd> x,
                                     Eigen::Ref<Eigen::VectorXd> xd)
{
  set_y_init(y_init);
  integrateStart(x, xd);
}

void DynamicalSystem::integrateStart(Eigen::Ref<Eigen::VectorXd> x,
                                     Eigen::Ref<Eigen::VectorXd> xd) const
{
  x = x_init_;
  differentialEquation(x, xd);
}

void DynamicalSystem::integrateStepEuler(double dt, const Ref<const VectorXd> x,
                                         Ref<VectorXd> x_updated,
                                         Ref<VectorXd> xd_updated) const
{
  assert(dt > 0.0);
  assert(x.size() == dim_x_);

  ENTERING_REAL_TIME_CRITICAL_CODE
  differentialEquation(x, xd_updated);
  x_updated = x + dt * xd_updated;  // Euler integration
  EXITING_REAL_TIME_CRITICAL_CODE
}

void DynamicalSystem::integrateStepRungeKutta(double dt,
                                              const Ref<const VectorXd> x,
                                              Ref<VectorXd> x_updated,
                                              Ref<VectorXd> xd_updated) const
{
  assert(dt > 0.0);
  assert(x.size() == dim_x_);

  ENTERING_REAL_TIME_CRITICAL_CODE

  // 4th order Runge-Kutta for a 1st order system
  // http://en.wikipedia.org/wiki/Runge-Kutta_method#The_Runge.E2.80.93Kutta_method
  differentialEquation(x, k1_);
  input_k2_ = x + dt * 0.5 * k1_;
  differentialEquation(input_k2_, k2_);
  input_k3_ = x + dt * 0.5 * k2_;
  differentialEquation(input_k3_, k3_);
  input_k4_ = x + dt * k3_;
  differentialEquation(input_k4_, k4_);

  x_updated = x + dt * (k1_ + 2.0 * (k2_ + k3_) + k4_) / 6.0;
  differentialEquation(x_updated, xd_updated);

  EXITING_REAL_TIME_CRITICAL_CODE
}

void from_json(const nlohmann::json& j, DynamicalSystem*& obj)
{
  string class_name = j.at("py/object").get<string>();

  if (class_name.find("ExponentialSystem") != string::npos) {
    obj = j.get<ExponentialSystem*>();

  } else if (class_name.find("SigmoidSystem") != string::npos) {
    obj = j.get<SigmoidSystem*>();

  } else if (class_name.find("SpringDamperSystem") != string::npos) {
    obj = j.get<SpringDamperSystem*>();

  } else if (class_name.find("TimeSystem") != string::npos) {
    obj = j.get<TimeSystem*>();

  } else {
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "Unknown DynamicalSystem: " << class_name << endl;
    obj = NULL;
  }
}

void DynamicalSystem::to_json_base(nlohmann::json& j) const
{
  j["_dim_x"] = dim_x_;
  j["_dim_y"] = dim_y_;
  j["_tau"] = tau_;
  j["_y_init"] = x_init_.segment(0, dim_y_);

  string c("DynamicalSystem");
  j["py/object"] = "dynamicalsystems." + c + "." + c;  // for jsonpickle
}

std::ostream& operator<<(std::ostream& output, const DynamicalSystem& d)
{
  nlohmann::json j;
  d.to_json_helper(j);
  // output << j.dump(4);
  output << j.dump();
  return output;
}

}  // namespace DmpBbo
