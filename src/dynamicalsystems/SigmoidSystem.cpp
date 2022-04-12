/**
 * @file   SigmoidSystem.cpp
 * @brief  SigmoidSystem class source file.
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

#include "dynamicalsystems/SigmoidSystem.hpp"

#include <eigen3/Eigen/Core>
#include <nlohmann/json.hpp>

#include "eigenutils/eigen_json.hpp"
#include "eigenutils/eigen_realtime_check.hpp"

using namespace std;
using namespace Eigen;

namespace DmpBbo {

SigmoidSystem::SigmoidSystem(double tau, const Eigen::VectorXd& x_init,
                             double max_rate, double inflection_point_time)
    : DynamicalSystem(1, tau, x_init),
      max_rate_(max_rate),
      inflection_point_time_(inflection_point_time)
{
  Ks_ = SigmoidSystem::computeKs(x_init, max_rate_, inflection_point_time_);
}

SigmoidSystem::~SigmoidSystem(void) {}

void SigmoidSystem::set_tau(double new_tau)
{
  // Get previous tau from superclass with tau() and set it with set_tau()
  double prev_tau = tau();
  DynamicalSystem::set_tau(new_tau);

  inflection_point_time_ =
      new_tau * inflection_point_time_ / prev_tau;  // todo document this
  Ks_ = SigmoidSystem::computeKs(x_init(), max_rate_, inflection_point_time_);
}

void SigmoidSystem::set_x_init(const VectorXd& x_init)
{
  assert(x_init.size() == dim());
  DynamicalSystem::set_x_init(x_init);
  Ks_ = SigmoidSystem::computeKs(x_init, max_rate_, inflection_point_time_);
}

VectorXd SigmoidSystem::computeKs(const VectorXd& N_0s, double r,
                                  double inflection_point_time_time)
{
  // The idea here is that the initial state (called N_0s above), max_rate (r
  // above) and the inflection_point_time are set by the user. The only
  // parameter that we have left to tune is the "carrying capacity" K.
  //   http://en.wikipedia.org/wiki/Logistic_function#In_ecology:_modeling_population_growth
  // In the below, we set K so that the initial state is N_0s for the given r
  // and tau

  // Known
  //   N(t) = K / ( 1 + (K/N_0 - 1)*exp(-r*t))
  //   N(t_inf) = K / 2
  // Plug into each other and solve for K
  //   K / ( 1 + (K/N_0 - 1)*exp(-r*t_infl)) = K/2
  //              (K/N_0 - 1)*exp(-r*t_infl) = 1
  //                             (K/N_0 - 1) = 1/exp(-r*t_infl)
  //                                       K = N_0*(1+(1/exp(-r*t_infl)))
  VectorXd Ks = N_0s;
  for (int dd = 0; dd < Ks.size(); dd++)
    Ks[dd] = N_0s[dd] * (1 + (1 / exp(-r * inflection_point_time_time)));

  // If Ks is too close to N_0===initial_state, then the differential equation
  // will always return 0 See differentialEquation below
  //   xd = max_rate_*x*(1-(x/Ks_));
  // For initial_state this is
  //   xd = max_rate_*initial_state*(1-(initial_state/Ks_));
  // If initial_state is very close/equal to Ks we get
  //   xd = max_rate_*Ks*(1-(Ks/Ks_));
  //   xd = max_rate_*Ks*(1-1);
  //   xd = max_rate_*Ks*0;
  //   xd = 0;
  // And integration fails, especially for Euler integration.
  // So we now give a warning if this is likely to happen.
  VectorXd div = N_0s.array() / Ks.array() - 1;
  // cout << setprecision(20) <<  div << endl;
  if ((div.array().abs() < 10e-9).any())  // 10e-9 determined empirically
  {
    cerr << endl << __FILE__ << ":" << __LINE__ << ":";
    cerr << "In function SigmoidSystem::computeKs(), Ks is too close to N_0s. "
            "This may lead to errors during numerical integration. Recommended "
            "solution: choose a lower magnitude for the maximum rate of change "
            "(currently it is "
         << r << ")" << endl;
  }

  return Ks;
}

void SigmoidSystem::differentialEquation(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    Eigen::Ref<Eigen::VectorXd> xd) const
{
  ENTERING_REAL_TIME_CRITICAL_CODE

  xd = max_rate_ * x.array() * (1 - (x.array() / Ks_.array()));

  EXITING_REAL_TIME_CRITICAL_CODE
}

void SigmoidSystem::analyticalSolution(const VectorXd& ts, MatrixXd& xs,
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

  // Auxillary variables to improve legibility
  double r = max_rate_;
  VectorXd exp_rt = (-r * ts).array().exp();

  VectorXd y_init = x_init();

  for (int dd = 0; dd < dim(); dd++) {
    // Auxillary variables to improve legibility
    double K = Ks_[dd];
    double b = (K / y_init[dd]) - 1;

    xs.block(0, dd, n_time_steps, 1) = K / (1 + b * exp_rt.array());
    xds.block(0, dd, n_time_steps, 1) =
        K * r * b * ((1 + b * exp_rt.array()).square().inverse().array()) *
        exp_rt.array();
  }

  if (caller_expects_transposed) {
    xs.transposeInPlace();
    xds.transposeInPlace();
  }
}

void from_json(const nlohmann::json& j, SigmoidSystem*& obj)
{
  double tau = from_json_to_double(j.at("tau_"));
  double max_rate = from_json_to_double(j.at("max_rate_"));
  double inflection_point_time =
      from_json_to_double(j.at("inflection_point_time_"));
  VectorXd y_init = j.at("initial_state_").at("values");

  obj = new SigmoidSystem(tau, y_init, max_rate, inflection_point_time);
}

void SigmoidSystem::to_json_helper(nlohmann::json& j) const
{
  to_json_base(j);  // Get the json string from the base class

  j["max_rate_"] = max_rate_;
  j["inflection_point_time_"] = inflection_point_time_;

  string c("SigmoidSystem");
  j["py/object"] = "dynamicalsystems." + c + "." + c;  // for jsonpickle
}

}  // namespace DmpBbo
