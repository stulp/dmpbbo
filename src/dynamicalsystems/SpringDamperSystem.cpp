/**
 * @file   SpringDamperSystem.cpp
 * @brief  SpringDamperSystem class source file.
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

#include "dynamicalsystems/SpringDamperSystem.hpp"

#include <eigen3/Eigen/Core>
#include <nlohmann/json.hpp>

#include "eigenutils/eigen_json.hpp"
#include "eigenutils/eigen_realtime_check.hpp"

using namespace std;
using namespace Eigen;

namespace DmpBbo {

SpringDamperSystem::SpringDamperSystem(double tau, Eigen::VectorXd y_init,
                                       Eigen::VectorXd y_attr,
                                       double damping_coefficient,
                                       double spring_constant, double mass)
    : DynamicalSystem(2, tau, y_init),
      y_attr_(y_attr),
      damping_coefficient_(damping_coefficient),
      spring_constant_(spring_constant),
      mass_(mass)
{
  if (spring_constant_ == CRITICALLY_DAMPED)
    spring_constant_ =
        damping_coefficient_ * damping_coefficient_ / 4;  // Critically damped

  // Order 2 system: these variables are half the size of state dimensionality
  y_ = VectorXd(dim() / 2);
  z_ = VectorXd(dim() / 2);
  yd_ = VectorXd(dim() / 2);
  zd_ = VectorXd(dim() / 2);
}

SpringDamperSystem::~SpringDamperSystem(void) {}

void SpringDamperSystem::differentialEquation(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    Eigen::Ref<Eigen::VectorXd> xd) const
{
  ENTERING_REAL_TIME_CRITICAL_CODE

  // Spring-damper system was originally 2nd order, i.e. with [x xd xdd]
  // After rewriting it as a 1st order system it becomes [y z yd zd], with yd =
  // z;

  // Get 'y' and 'z' parts of the state in 'x'
  // (These memory for these vectors has been initialized in the constructor to
  // enable realtime.)
  int y_dim = dim() / 2;
  y_ = x.segment(0, y_dim);
  z_ = x.segment(y_dim, y_dim);

  // Compute yd and zd. See
  // http://en.wikipedia.org/wiki/Damped_spring-mass_system#Example:mass_.E2.80.93spring.E2.80.93damper
  // and equation 2.1 of
  // http://www-clmc.usc.edu/publications/I/ijspeert-NC2013.pdf

  yd_ = z_ / tau();

  zd_ = (-spring_constant_ * (y_ - y_attr_) - damping_coefficient_ * z_) /
        (mass_ * tau());

  xd.segment(0, dim() / 2) = yd_;
  xd.segment(dim() / 2, dim() / 2) = zd_;

  EXITING_REAL_TIME_CRITICAL_CODE
}

void SpringDamperSystem::analyticalSolution(const VectorXd& ts, MatrixXd& xs,
                                            MatrixXd& xds) const
{
  int n_time_steps = ts.size();

  // Usually, we expect xs and xds to be of size n_time_steps X dim, so we
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

  VectorXd x_ini = x_init();

  // Closed form solution to 2nd order canonical system
  // This system behaves like a critically damped spring-damper system
  // http://en.wikipedia.org/wiki/Damped_spring-mass_system
  double omega_0 = sqrt(spring_constant_ / mass_) / tau();  // natural frequency
  double zeta = damping_coefficient_ /
                (2 * sqrt(mass_ * spring_constant_));  // damping ratio
  if (zeta != 1.0)
    cout << "WARNING: Spring-damper system is not critically damped, zeta="
         << zeta << endl;

  // The loop is slower, but more legible than fudging around with
  // Eigen::replicate().
  for (int i_dim = 0; i_dim < dim_y(); i_dim++) {
    double y0 = x_ini[i_dim] - y_attr_[i_dim];
    double yd0 = x_ini[dim_y() + i_dim];

    double A = y0;
    double B = yd0 + omega_0 * y0;

    // Closed form solutions
    // See http://en.wikipedia.org/wiki/Damped_spring-mass_system
    VectorXd exp_term = -omega_0 * ts;
    exp_term = exp_term.array().exp();

    int Y = 0 * dim_y() + i_dim;
    int Z = 1 * dim_y() + i_dim;

    VectorXd ABts = A + B * ts.array();

    // Closed form solutions
    // See http://en.wikipedia.org/wiki/Damped_spring-mass_system
    xs.col(Y) = y_attr_(i_dim) + ((ABts.array())) * exp_term.array();

    // Derivative of the above (use product rule: (f*g)' = f'*g + f*g'
    xds.col(Y) = ((B - omega_0 * ABts.array())) * exp_term.array();

    // Derivative of the above (again use product    rule: (f*g)' = f'*g + f*g'
    VectorXd ydds =
        (-omega_0 * (2 * B - omega_0 * ABts.array())) * exp_term.array();

    // This is how to compute the 'z' terms from the 'y' terms
    xs.col(Z) = xds.col(Y) * tau();
    xds.col(Z) = ydds * tau();
  }

  if (caller_expects_transposed) {
    xs.transposeInPlace();
    xds.transposeInPlace();
  }
}

void from_json(const nlohmann::json& j, SpringDamperSystem*& obj)
{
  double tau = j.at("_tau");
  double damping_coefficient = j.at("_damping_coefficient");
  double spring_constant = j.at("_spring_constant");
  double mass = j.at("_mass");

  VectorXd y_attr = j.at("_y_attr");
  VectorXd x_init = j.at("_x_init");
  VectorXd y_init = x_init.segment(0, y_attr.size());

  obj = new SpringDamperSystem(tau, y_init, y_attr, damping_coefficient,
                               spring_constant, mass);
}

void SpringDamperSystem::to_json_helper(nlohmann::json& j) const
{
  to_json_base(j);  // Get the json string from the base class

  j["_damping_coefficient"] = damping_coefficient_;
  j["_spring_constant"] = spring_constant_;
  j["_mass"] = mass_;
  j["_y_attr"] = y_attr_;
  j["class"] = "SpringDamperSystem";
}

}  // namespace DmpBbo
