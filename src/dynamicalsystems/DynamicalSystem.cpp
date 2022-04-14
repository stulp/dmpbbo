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

DynamicalSystem::DynamicalSystem(int order, double tau,
                                 Eigen::VectorXd xy_init)
    :  // For 1st order systems, the dimensionality of the state vector 'x' is
       // 'dim' For 2nd order systems, the system is expanded to x = [y z],
       // where 'y' and 'z' are both of dimensionality 'dim'. Therefore dim(x)
       // is 2*dim
      dim_(xy_init.size() * order),
      tau_(tau)
{
  assert(order == 1 || order == 2);

  if (order == 1) {
    x_init_ = xy_init;
  } else {  // order = 2
    // 2nd order: expand the state, and fill it with zeros.
    x_init_ = VectorXd::Zero(dim_);
    x_init_.segment(0, xy_init.size()) = xy_init;
  }

  preallocateMemory(dim_);
}

DynamicalSystem::DynamicalSystem(double tau, Eigen::VectorXd y_init, int n_dims)
    : dim_(n_dims), tau_(tau)
{
  x_init_ = VectorXd::Zero(dim_);
  x_init_.segment(0, y_init.size()) = y_init;
  preallocateMemory(dim_);
}

void DynamicalSystem::preallocateMemory(int dim)
{
  // Pre-allocate memory for Runge-Kutta integration
  k1_ = VectorXd(dim);
  k2_ = VectorXd(dim);
  k3_ = VectorXd(dim);
  k4_ = VectorXd(dim);

  input_k2_ = VectorXd(dim);
  input_k3_ = VectorXd(dim);
  input_k4_ = VectorXd(dim);
}

DynamicalSystem::~DynamicalSystem(void) {}

void DynamicalSystem::set_x_init(const Eigen::VectorXd& xy_init)
{
  if (xy_init.size() == dim_) {
    // Standard 1st order system
    x_init_ = xy_init;
  } else {
    // All other cases: pad with zeros
    x_init_.fill(0.0);
    x_init_.segment(0, xy_init.size()) = xy_init;
  }
}

void DynamicalSystem::integrateStart(const Eigen::VectorXd& xy_init,
                                     Eigen::Ref<Eigen::VectorXd> x,
                                     Eigen::Ref<Eigen::VectorXd> xd)
{
  set_x_init(xy_init);
  integrateStart(x, xd);
}

void DynamicalSystem::integrateStart(Eigen::Ref<Eigen::VectorXd> x,
                                     Eigen::Ref<Eigen::VectorXd> xd) const
{
  // Check size. Leads to faster numerical integration and makes Eigen::Ref
  // easier to deal with
  assert(x.size() == dim_);
  assert(xd.size() == dim_);

  x = x_init_;
  differentialEquation(x, xd);
}

void DynamicalSystem::integrateStepEuler(double dt, const Ref<const VectorXd> x,
                                         Ref<VectorXd> x_updated,
                                         Ref<VectorXd> xd_updated) const
{
  assert(dt > 0.0);
  assert(x.size() == dim_);

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
  assert(x.size() == dim_);

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
  j["dim_"] = dim_;
  j["tau_"] = tau_;
  j["initial_state_"] = x_init_;

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
