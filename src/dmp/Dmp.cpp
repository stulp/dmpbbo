/**
 * @file Dmp.cpp
 * @brief  Dmp class source file.
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

#include "dmp/Dmp.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

#include "dmp/Trajectory.hpp"
#include "dynamicalsystems/DynamicalSystem.hpp"
#include "dynamicalsystems/ExponentialSystem.hpp"
#include "dynamicalsystems/SigmoidSystem.hpp"
#include "dynamicalsystems/SpringDamperSystem.hpp"
#include "dynamicalsystems/TimeSystem.hpp"
#include "eigenutils/eigen_file_io.hpp"
#include "eigenutils/eigen_json.hpp"
#include "eigenutils/eigen_realtime_check.hpp"
#include "functionapproximators/FunctionApproximator.hpp"

using namespace std;
using namespace Eigen;

namespace DmpBbo {

/** Extracts all variables of the spring-damper system from a state vector, e.g.
 * state.SPRING */
#define SPRING segment(0 * dim_y() + 0, 2 * dim_y())
/** Extracts first order variables of the spring-damper system from a state
 * vector, e.g. state.SPRINGM_Y */
#define SPRING_Y segment(0 * dim_y() + 0, dim_y())
/** Extracts second order variables of the spring-damper system from a state
 * vector, e.g. state.SPRING_Z */
#define SPRING_Z segment(1 * dim_y() + 0, dim_y())
/** Extracts all variables of the goal from a state vector, e.g. state.GOAL */
#define GOAL segment(2 * dim_y() + 0, dim_y())
/** Extracts the phase variable (1-D) from a state vector, e.g. state.PHASE */
#define PHASE segment(3 * dim_y() + 0, 1)
/** Extracts all variables of the gating system from a state vector, e.g.
 * state.GATING */
#define GATING segment(3 * dim_y() + 1, 1)

/** Extracts first T (time steps) state vectors of the spring-damper system ,
 * e.g. states.SPRING(100) */
#define SPRINGM(T) block(0, 0 * dim_y() + 0, T, 2 * dim_y())
/** Extracts first T (time steps) state vectors of the spring-damper system ,
 * e.g. states.SPRINGM_Y(100) */
#define SPRINGM_Y(T) block(0, 0 * dim_y() + 0, T, dim_y())
/** Extracts first T (time steps) state vectors of the spring-damper system ,
 * e.g. states.SPRINGM_Z(100) */
#define SPRINGM_Z(T) block(0, 1 * dim_y() + 0, T, dim_y())
/** Extracts first T (time steps) state vectors of the goal system, e.g.
 * states.GOALM(100) */
#define GOALM(T) block(0, 2 * dim_y() + 0, T, dim_y())
/** Extracts first T (time steps) states of the phase system, e.g.
 * states.PHASEM(100) */
#define PHASEM(T) block(0, 3 * dim_y() + 0, T, 1)
/** Extracts first T (time steps) state vectors of the gating system, e.g.
 * states.GATINGM(100) */
#define GATINGM(T) block(0, 3 * dim_y() + 1, T, 1)

Dmp::Dmp(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr,
         std::vector<FunctionApproximator*> function_approximators,
         double alpha_spring_damper, ExponentialSystem* goal_system,
         DynamicalSystem* phase_system, DynamicalSystem* gating_system,
         std::string scaling, Eigen::VectorXd scaling_amplitudes)
    : DynamicalSystem(tau, y_init, 3 * y_init.size() + 2),
      y_attr_(y_attr),
      goal_system_(goal_system),
      phase_system_(phase_system),
      gating_system_(gating_system),
      forcing_term_scaling_(scaling),
      scaling_amplitudes_(scaling_amplitudes)
{
  initSubSystems(alpha_spring_damper, goal_system, phase_system, gating_system);
  initFunctionApproximators(function_approximators);
}

Dmp::Dmp(int n_dims_dmp,
         std::vector<FunctionApproximator*> function_approximators,
         double alpha_spring_damper, ExponentialSystem* goal_system,
         DynamicalSystem* phase_system, DynamicalSystem* gating_system,
         std::string scaling, Eigen::VectorXd scaling_amplitudes)
    : DynamicalSystem(1, 1.0, VectorXd::Zero(n_dims_dmp)),
      y_attr_(VectorXd::Ones(n_dims_dmp)),
      goal_system_(goal_system),
      phase_system_(phase_system),
      gating_system_(gating_system),
      forcing_term_scaling_(scaling),
      scaling_amplitudes_(scaling_amplitudes)
{
  initSubSystems(alpha_spring_damper, goal_system, phase_system, gating_system);
  initFunctionApproximators(function_approximators);
}

Dmp::Dmp(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr,
         std::vector<FunctionApproximator*> function_approximators,
         std::string dmp_type, std::string scaling,
         Eigen::VectorXd scaling_amplitudes)
    : DynamicalSystem(1, tau, y_init),
      y_attr_(y_attr),
      forcing_term_scaling_(scaling),
      scaling_amplitudes_(scaling_amplitudes)
{
  initSubSystems(dmp_type);
  initFunctionApproximators(function_approximators);
}

Dmp::Dmp(int n_dims_dmp,
         std::vector<FunctionApproximator*> function_approximators,
         std::string dmp_type, std::string scaling,
         Eigen::VectorXd scaling_amplitudes)
    : DynamicalSystem(1, 1.0, VectorXd::Zero(n_dims_dmp)),
      y_attr_(VectorXd::Ones(n_dims_dmp)),
      forcing_term_scaling_(scaling),
      scaling_amplitudes_(scaling_amplitudes)
{
  initSubSystems(dmp_type);
  initFunctionApproximators(function_approximators);
}

Dmp::Dmp(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr,
         double alpha_spring_damper, ExponentialSystem* goal_system)
    : DynamicalSystem(1, tau, y_init),
      y_attr_(y_attr),
      forcing_term_scaling_("NO_SCALING"),
      scaling_amplitudes_(VectorXd::Zero(0))
{
  VectorXd one_1 = VectorXd::Ones(1);
  VectorXd one_0 = VectorXd::Zero(1);
  DynamicalSystem* phase_system = new ExponentialSystem(tau, one_1, one_0, 4);
  DynamicalSystem* gating_system = new ExponentialSystem(tau, one_1, one_0, 4);
  initSubSystems(alpha_spring_damper, goal_system, phase_system, gating_system);

  vector<FunctionApproximator*> function_approximators(y_init.size());
  for (int dd = 0; dd < y_init.size(); dd++) function_approximators[dd] = NULL;
  initFunctionApproximators(function_approximators);
}

void Dmp::initSubSystems(std::string dmp_type)
{
  VectorXd one_1 = VectorXd::Ones(1);
  VectorXd one_0 = VectorXd::Zero(1);

  ExponentialSystem* goal_system = NULL;
  DynamicalSystem* phase_system = NULL;
  DynamicalSystem* gating_system = NULL;
  if (dmp_type == "IJSPEERT_2002_MOVEMENT") {
    goal_system = NULL;
    phase_system = new ExponentialSystem(tau(), one_1, one_0, 4);
    gating_system = new ExponentialSystem(tau(), one_1, one_0, 4);
  } else if (dmp_type == "KULVICIUS_2012_JOINING" ||
             dmp_type == "COUNTDOWN_2013") {
    goal_system = new ExponentialSystem(tau(), y_init(), y_attr_, 15);
    gating_system = new SigmoidSystem(tau(), one_1, -10, 0.9 * tau());
    bool count_down = (dmp_type == "COUNTDOWN_2013");
    phase_system = new TimeSystem(tau(), count_down);
  }

  double alpha_spring_damper = 20;

  initSubSystems(alpha_spring_damper, goal_system, phase_system, gating_system);
}

void Dmp::initSubSystems(double alpha_spring_damper,
                         ExponentialSystem* goal_system,
                         DynamicalSystem* phase_system,
                         DynamicalSystem* gating_system)
{
  // Make room for the subsystems
  // dim_ = 3 * dim_y() + 2;

  spring_system_ =
      new SpringDamperSystem(tau(), y_init(), y_attr_, alpha_spring_damper);

  goal_system_ = goal_system;
  if (goal_system != NULL) {
    assert(goal_system->dim() == dim_y());
    // Initial state of the goal system is that same as that of the DMP
    goal_system_->set_x_init(y_init());
  }

  phase_system_ = phase_system;

  gating_system_ = gating_system;

  if (forcing_term_scaling_ == "AMPLITUDE_SCALING") {
    assert(scaling_amplitudes_.size() == dim_dmp());
  }

  // Pre-allocate memory for real-time execution
  y_init_prealloc_ = VectorXd(dim_y());
  fa_output_one_prealloc_ = VectorXd(1);
  fa_output_prealloc_ = MatrixXd(1, dim_y());
  forcing_term_prealloc_ = VectorXd(dim_y());
  g_minus_y0_prealloc_ = VectorXd(dim_y());
}

void Dmp::set_damping_coefficient(double damping_coefficient)
{
  spring_system_->set_damping_coefficient(damping_coefficient);
}
void Dmp::set_spring_constant(double spring_constant)
{
  spring_system_->set_spring_constant(spring_constant);
}

void Dmp::initFunctionApproximators(
    vector<FunctionApproximator*> function_approximators)
{
  if (function_approximators.empty()) return;

  assert(dim_y() == (int)function_approximators.size());

  function_approximators_ = function_approximators;
}

Dmp::~Dmp(void)
{
  delete goal_system_;
  delete spring_system_;
  delete phase_system_;
  delete gating_system_;
  for (unsigned int ff = 0; ff < function_approximators_.size(); ff++)
    delete (function_approximators_[ff]);
}

void Dmp::integrateStart(Ref<VectorXd> x, Ref<VectorXd> xd) const
{
  assert(x.size() == dim());
  assert(xd.size() == dim());

  x.fill(0);
  xd.fill(0);

  // Start integrating goal system if it exists
  if (goal_system_ == NULL) {
    // No goal system, simply set goal state to attractor state
    x.GOAL = y_attr_;
    xd.GOAL.fill(0);
  } else {
    // Goal system exists. Start integrating it.
    goal_system_->integrateStart(y_init(), x.GOAL, xd.GOAL);
  }

  // Set the attractor state of the spring system
  spring_system_->set_y_attr(x.GOAL);

  // Start integrating all futher subsystems
  spring_system_->integrateStart(y_init(), x.SPRING, xd.SPRING);
  phase_system_->integrateStart(x.PHASE, xd.PHASE);
  gating_system_->integrateStart(x.GATING, xd.GATING);

  // Add rates of change
  differentialEquation(x, xd);
}

void Dmp::differentialEquation(const Eigen::Ref<const Eigen::VectorXd>& x,
                               Eigen::Ref<Eigen::VectorXd> xd) const
{
  ENTERING_REAL_TIME_CRITICAL_CODE

  if (goal_system_ == NULL) {
    // If there is no dynamical system for the delayed goal, the goal is
    // simply the attractor state
    spring_system_->set_y_attr(y_attr_);
    // with zero change
    xd.GOAL.fill(0);
  } else {
    // Integrate goal system and get current goal state
    goal_system_->set_x_attr(y_attr_);
    goal_system_->differentialEquation(x.GOAL, xd.GOAL);
    // The goal state is the attractor state of the spring-damper system
    spring_system_->set_y_attr(x.GOAL);
  }

  // Integrate spring damper system
  // Forcing term is added to spring_state later
  spring_system_->differentialEquation(x.SPRING, xd.SPRING);

  // Non-linear forcing term
  phase_system_->differentialEquation(x.PHASE, xd.PHASE);
  gating_system_->differentialEquation(x.GATING, xd.GATING);

  // Compute output of the funciton approximators
  for (int i_dim = 0; i_dim < dim_y(); i_dim++) {
    function_approximators_[i_dim]->predictRealTime(x.PHASE,
                                                    fa_output_one_prealloc_);
    fa_output_prealloc_(0, i_dim) = fa_output_one_prealloc_(0, 0);
  }

  // Gate the output of the function approximators
  int t0 = 0;
  double gating = (x.GATING)[0];
  forcing_term_prealloc_ = gating * fa_output_prealloc_.row(t0);

  // Scale the forcing term, if necessary
  if (forcing_term_scaling_ == "G_MINUS_Y0_SCALING") {
    get_y_init(y_init_prealloc_);
    g_minus_y0_prealloc_ = (y_attr_ - y_init_prealloc_).transpose();
    forcing_term_prealloc_ =
        forcing_term_prealloc_.array() * g_minus_y0_prealloc_.array();
  } else if (forcing_term_scaling_ == "AMPLITUDE_SCALING") {
    forcing_term_prealloc_ =
        forcing_term_prealloc_.array() * scaling_amplitudes_.array();
  }

  // Add forcing term to the ZD component of the spring state
  xd.SPRING_Z = xd.SPRING_Z + forcing_term_prealloc_ / tau();

  EXITING_REAL_TIME_CRITICAL_CODE
}

void Dmp::statesAsTrajectory(const Eigen::MatrixXd& x_in,
                             const Eigen::MatrixXd& xd_in,
                             Eigen::MatrixXd& y_out, Eigen::MatrixXd& yd_out,
                             Eigen::MatrixXd& ydd_out) const
{
  int n_time_steps = x_in.rows();
  y_out = x_in.SPRINGM_Y(n_time_steps);
  yd_out = xd_in.SPRINGM_Y(n_time_steps);
  ydd_out = xd_in.SPRINGM_Z(n_time_steps) / tau();
  // MatrixXd z_out, zd_out;
  // z_out  = x_in.SPRINGM_Z(n_time_steps);
  // zd_out = xd_in.SPRINGM_Z(n_time_steps);
  // Divide by tau to go from z to y space
  // yd = z_out/obj.tau;
  // ydd_out = zd_out/tau();
}

void Dmp::statesAsTrajectory(const Eigen::VectorXd& ts,
                             const Eigen::MatrixXd& x_in,
                             const Eigen::MatrixXd& xd_in,
                             Trajectory& trajectory) const
{
  int n_time_steps = ts.rows();
#ifndef NDEBUG  // Variables below are only required for asserts; check for
                // NDEBUG to avoid warnings.
  int n_dims = x_in.cols();
#endif
  assert(n_time_steps == x_in.rows());
  assert(n_time_steps == xd_in.rows());
  assert(n_dims == xd_in.cols());

  // Left column is time
  Trajectory new_trajectory(ts,
                            // y_out (see function above)
                            x_in.SPRINGM_Y(n_time_steps),
                            // yd_out (see function above)
                            xd_in.SPRINGM_Y(n_time_steps),
                            // ydd_out (see function above)
                            xd_in.SPRINGM_Z(n_time_steps) / tau());

  trajectory = new_trajectory;
}

void Dmp::analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs,
                             Eigen::MatrixXd& xds,
                             Eigen::MatrixXd& forcing_terms,
                             Eigen::MatrixXd& fa_outputs) const
{
  int n_time_steps = ts.size();
  assert(n_time_steps > 0);

  // Usually, we expect xs and xds to be of size T X dim(), so we resize to
  // that. However, if the input matrices were of size dim() X T, we return the
  // matrices of that size by doing a transposeInPlace at the end. That way, the
  // user can also request dim() X T sized matrices.
  bool caller_expects_transposed =
      (xs.rows() == dim() && xs.cols() == n_time_steps);

  // INTEGRATE SYSTEMS ANALYTICALLY AS MUCH AS POSSIBLE

  // Integrate phase
  MatrixXd xs_phase;
  MatrixXd xds_phase;
  phase_system_->analyticalSolution(ts, xs_phase, xds_phase);

  // Compute gating term
  MatrixXd xs_gating;
  MatrixXd xds_gating;
  gating_system_->analyticalSolution(ts, xs_gating, xds_gating);

  // Compute the output of the function approximator
  fa_outputs.resize(ts.size(), dim_y());
  fa_outputs.fill(0.0);

  MatrixXd fa_output_one_dim(n_time_steps, 1);
  for (int i_dim = 0; i_dim < dim_y(); i_dim++) {
    function_approximators_[i_dim]->predict(xs_phase, fa_output_one_dim);
    fa_outputs.col(i_dim) = fa_output_one_dim;
  }

  // Gate the output to get the forcing ter4m
  MatrixXd xs_gating_rep = xs_gating.replicate(1, fa_outputs.cols());
  forcing_terms = fa_outputs.array() * xs_gating_rep.array();

  // Scale the forcing term, if necessary
  if (forcing_term_scaling_ == "G_MINUS_Y0_SCALING") {
    MatrixXd g_minus_y0_rep =
        (y_attr_ - y_init()).transpose().replicate(n_time_steps, 1);
    forcing_terms = forcing_terms.array() * g_minus_y0_rep.array();
  } else if (forcing_term_scaling_ == "AMPLITUDE_SCALING") {
    MatrixXd scaling_amplitudes_rep =
        scaling_amplitudes_.transpose().replicate(n_time_steps, 1);
    forcing_terms = forcing_terms.array() * scaling_amplitudes_rep.array();
  }

  MatrixXd xs_goal, xds_goal;
  // Get current delayed goal
  if (goal_system_ == NULL) {
    // If there is no dynamical system for the delayed goal, the goal is
    // simply the attractor state
    xs_goal = y_attr_.transpose().replicate(n_time_steps, 1);
    // with zero change
    xds_goal = MatrixXd::Zero(n_time_steps, dim_y());
  } else {
    // Integrate goal system and get current goal state
    goal_system_->analyticalSolution(ts, xs_goal, xds_goal);
  }

  xs.resize(n_time_steps, dim());
  xds.resize(n_time_steps, dim());

  int T = n_time_steps;

  xs.GOALM(T) = xs_goal;
  xds.GOALM(T) = xds_goal;
  xs.PHASEM(T) = xs_phase;
  xds.PHASEM(T) = xds_phase;
  xs.GATINGM(T) = xs_gating;
  xds.GATINGM(T) = xds_gating;

  // THE REST CANNOT BE DONE ANALYTICALLY

  // Reset the dynamical system, and get the first state
  double damping = spring_system_->damping_coefficient();
  SpringDamperSystem localspring_system_(tau(), y_init(), y_attr_, damping);

  // Set first attractor state
  localspring_system_.set_y_attr(xs_goal.row(0));

  // Start integrating spring damper system
  int dim_spring = localspring_system_.dim();
  VectorXd x_spring(dim_spring),
      xd_spring(dim_spring);  // todo Why are these needed?
  int t0 = 0;
  localspring_system_.integrateStart(x_spring, xd_spring);
  xs.row(t0).SPRING = x_spring;
  xds.row(t0).SPRING = xd_spring;

  // Add forcing term to the acceleration of the spring state
  xds.SPRINGM_Z(1) = xds.SPRINGM_Z(1) + forcing_terms.row(t0) / tau();

  for (int tt = 1; tt < n_time_steps; tt++) {
    double dt = ts[tt] - ts[tt - 1];

    // Euler integration
    xs.row(tt).SPRING = xs.row(tt - 1).SPRING + dt * xds.row(tt - 1).SPRING;

    // Set the attractor state of the spring system
    localspring_system_.set_y_attr(xs.row(tt).GOAL);

    // Integrate spring damper system
    localspring_system_.differentialEquation(xs.row(tt).SPRING, xd_spring);
    xds.row(tt).SPRING = xd_spring;

    // Add forcing term to the acceleration of the spring state
    xds.row(tt).SPRING_Z = xds.row(tt).SPRING_Z + forcing_terms.row(tt) / tau();
    // Compute y component from z
    xds.row(tt).SPRING_Y = xs.row(tt).SPRING_Z / tau();
  }

  if (caller_expects_transposed) {
    xs.transposeInPlace();
    xds.transposeInPlace();
  }
}

void Dmp::set_tau(double tau)
{
  DynamicalSystem::set_tau(tau);

  // Set value in all relevant subsystems also
  spring_system_->set_tau(tau);
  if (goal_system_ != NULL) goal_system_->set_tau(tau);
  phase_system_->set_tau(tau);
  gating_system_->set_tau(tau);
}

void Dmp::set_y_init(const Eigen::VectorXd& y_init)
{
  assert(y_init.size() == dim_y());
  DynamicalSystem::set_y_init(y_init);

  // Set value in all relevant subsystems also
  spring_system_->set_y_init(y_init);
  if (goal_system_ != NULL) goal_system_->set_x_init(y_init);
}

void Dmp::set_y_attr(const VectorXd& y_attr)
{
  y_attr_ = y_attr;

  // Set value in all relevant subsystems also
  if (goal_system_ != NULL) goal_system_->set_x_attr(y_attr);

  // Do NOT do the following. The attractor state of the spring system is
  // determined by the goal system spring_system_->set_y_attr(y_attr);
}

void from_json(const nlohmann::json& j, Dmp*& obj)
{
  double tau = j.at("_tau");

  double alpha_spring_damper = j.at("_spring_system").at("damping_coefficient");

  VectorXd y_init = j.at("_y_init");
  VectorXd y_attr = j.at("_y_attr");
  // from_json(j.at("_y_init"), y_init);
  // from_json(j.at("_y_attr"), y_attr);

  ExponentialSystem* goal_system;
  DynamicalSystem *phase_system, *gating_system;
  goal_system = j.at("_goal_system").get<ExponentialSystem*>();
  phase_system = j.at("_phase_system").get<DynamicalSystem*>();
  gating_system = j.at("_gating_system").get<DynamicalSystem*>();

  string forcing_term_scaling = j.at("_forcing_term_scaling");
  VectorXd scaling_amplitudes = j.at("_scaling_amplitudes");

  int n_dims = y_attr.size();
  vector<FunctionApproximator*> function_approximators;
  const auto& jrow = j.at("_function_approximators");
  if (jrow.is_array()) {
    for (int i_dim = 0; i_dim < n_dims; i_dim++) {
      FunctionApproximator* fa = jrow.at(i_dim).get<FunctionApproximator*>();
      function_approximators.push_back(fa);
    }
  }

  obj = new Dmp(tau, y_init, y_attr, function_approximators,
                alpha_spring_damper, goal_system, phase_system, gating_system,
                forcing_term_scaling, scaling_amplitudes);
}

void Dmp::to_json_helper(nlohmann::json& j) const
{
  to_json_base(j);  // Get the json string from the base class

  j["_spring_system"]["damping_coefficient"] =
      spring_system_->damping_coefficient();
  j["_goal_system"] = goal_system_;
  j["_phase_system"] = phase_system_;
  j["_gating_system"] = gating_system_;
  j["_forcing_term_scaling"] = forcing_term_scaling_;
  j["_function_approximators"] = function_approximators_;
  j["class"] = "Dmp";
}

}  // namespace DmpBbo
