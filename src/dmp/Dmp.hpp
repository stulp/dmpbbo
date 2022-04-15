/**
 * @file Dmp.hpp
 * @brief  Dmp class header file.
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

#ifndef _DMP_H_
#define _DMP_H_

#define EIGEN_RUNTIME_NO_MALLOC  // Enable runtime tests for allocations

#include <eigen3/Eigen/Core>
#include <nlohmann/json_fwd.hpp>

#include "dynamicalsystems/DynamicalSystem.hpp"

namespace DmpBbo {

// forward declaration
class FunctionApproximator;
class ExponentialSystem;
class SpringDamperSystem;
class Trajectory;

/** \defgroup Dmps Dynamic Movement Primitives Module
 */

/**
 * \brief Implementation of Dynamical Movement Primitives.
 * \ingroup Dmps
 */
class Dmp : public DynamicalSystem {
 public:
  /**
   *  Initialization constructor.
   *  \param tau             Time constant
   *  \param y_init          Initial state
   *  \param y_attr          Attractor state
   *  \param alpha_spring_damper \f$\alpha\f$ in the spring-damper system of the
   * dmp \param goal_system     Dynamical system to compute delayed goal \param
   * phase_system    Dynamical system to compute the phase \param gating_system
   * Dynamical system to compute the gating term \param function_approximators
   * Function approximators for the forcing term \param scaling         Which
   * method to use for scaling the forcing term ("NO_SCALING",
   * "G_MINUS_Y0_SCALING", "AMPLITUDE_SCALING")
   */
  Dmp(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr,
      std::vector<FunctionApproximator*> function_approximators,
      double alpha_spring_damper, ExponentialSystem* goal_system,
      DynamicalSystem* phase_system, DynamicalSystem* gating_system,
      std::string scaling = "NO_SCALING");

  /**
   *  Initialization constructor for Dmps of known dimensionality, but with
   * unknown initial and attractor states. \param n_dims_dmp      Dimensionality
   * of the DMP \param alpha_spring_damper \f$\alpha\f$ in the spring-damper
   * system of the dmp \param goal_system     Dynamical system to compute
   * delayed goal \param phase_system    Dynamical system to compute the phase
   *  \param gating_system   Dynamical system to compute the gating term
   *  \param function_approximators Function approximators for the forcing term
   *  \param scaling         Which method to use for scaling the forcing term
   * ("NO_SCALING", "G_MINUS_Y0_SCALING", "AMPLITUDE_SCALING")
   */
  Dmp(int n_dims_dmp, std::vector<FunctionApproximator*> function_approximators,
      double alpha_spring_damper, ExponentialSystem* goal_system,
      DynamicalSystem* phase_system, DynamicalSystem* gating_system,
      std::string scaling = "NO_SCALING");

  /**
   *  Constructor that initializes the DMP with default dynamical systems.
   *  \param tau       Time constant
   *  \param y_init    Initial state
   *  \param y_attr    Attractor state
   *  \param function_approximators Function approximators for the forcing term
   *  \param dmp_type  The type of DMP ("IJSPEERT_2002_MOVEMENT",
   * "KULVICIUS_2012_JOINING", "COUNTDOWN_2013") \param scaling         Which
   * method to use for scaling the forcing term ("NO_SCALING",
   * "G_MINUS_Y0_SCALING", "AMPLITUDE_SCALING")
   */
  Dmp(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr,
      std::vector<FunctionApproximator*> function_approximators,
      std::string dmp_type = "KULVICIUS_2012_JOINING",
      std::string scaling = "NO_SCALING");

  /**
   *  Initialization constructor for Dmps of known dimensionality, but with
   * unknown initial and attractor states. Initializes the DMP with default
   * dynamical systems. \param n_dims_dmp      Dimensionality of the DMP \param
   * function_approximators Function approximators for the forcing term \param
   * dmp_type  The type of DMP ("IJSPEERT_2002_MOVEMENT",
   * "KULVICIUS_2012_JOINING", "COUNTDOWN_2013") \param scaling         Which
   * method to use for scaling the forcing term ("NO_SCALING",
   * "G_MINUS_Y0_SCALING", "AMPLITUDE_SCALING")
   */
  Dmp(int n_dims_dmp, std::vector<FunctionApproximator*> function_approximators,
      std::string dmp_type = "KULVICIUS_2012_JOINING",
      std::string scaling = "NO_SCALING");

  /**
   *  Initialization constructor for Dmps without a forcing term.
   *  \param tau             Time constant
   *  \param y_init          Initial state
   *  \param y_attr          Attractor state
   *  \param alpha_spring_damper \f$\alpha\f$ in the spring-damper system of the
   * dmp \param goal_system     Dynamical system to compute delayed goal
   */
  Dmp(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr,
      double alpha_spring_damper, ExponentialSystem* goal_system);

  /** Destructor. */
  ~Dmp(void);

  virtual void integrateStart(Eigen::Ref<Eigen::VectorXd> x,
                              Eigen::Ref<Eigen::VectorXd> xd) const;

  void differentialEquation(const Eigen::Ref<const Eigen::VectorXd>& x,
                            Eigen::Ref<Eigen::VectorXd> xd) const;

  /**
   * Return analytical solution of the system at certain times (and return
   * forcing terms)
   *
   * \param[in]  ts  A vector of times for which to compute the analytical
   * solutions \param[out] xs  Sequence of state vectors. T x D or D x T matrix,
   * where T is the number of times (the length of 'ts'), and D the size of the
   * state (i.e. dim_) \param[out] xds Sequence of state vectors (rates of
   * change). T x D or D x T matrix, where T is the number of times (the length
   * of 'ts'), and D the size of the state (i.e. dim_) \param[out]
   * forcing_terms The forcing terms for each dimension, for debugging purposes
   * only. \param[out] fa_output The output of the function approximators, for
   * debugging purposes only.
   *
   * \remarks The output xs and xds will be of size D x T \em only if the matrix
   * x you pass as an argument of size D x T. In all other cases (i.e. including
   * passing an empty matrix) the size of x will be T x D. This feature has been
   * added so that you may pass matrices of either size.
   */
  void analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs,
                          Eigen::MatrixXd& xds, Eigen::MatrixXd& forcing_terms,
                          Eigen::MatrixXd& fa_output) const;

  /**
   * Return analytical solution of the system at certain times (and return
   * forcing terms)
   *
   * \param[in]  ts  A vector of times for which to compute the analytical
   * solutions \param[out] xs  Sequence of state vectors. T x D or D x T matrix,
   * where T is the number of times (the length of 'ts'), and D the size of the
   * state (i.e. dim_) \param[out] xds Sequence of state vectors (rates of
   * change). T x D or D x T matrix, where T is the number of times (the length
   * of 'ts'), and D the size of the state (i.e. dim_) \param[out]
   * forcing_terms The forcing terms for each dimension, for debugging purposes
   * only.
   *
   * \remarks The output xs and xds will be of size D x T \em only if the matrix
   * x you pass as an argument of size D x T. In all other cases (i.e. including
   * passing an empty matrix) the size of x will be T x D. This feature has been
   * added so that you may pass matrices of either size.
   */
  inline void analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs,
                                 Eigen::MatrixXd& xds,
                                 Eigen::MatrixXd& forcing_terms) const
  {
    Eigen::MatrixXd fa_output;
    analyticalSolution(ts, xs, xds, forcing_terms, fa_output);
  }

  inline void analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs,
                                 Eigen::MatrixXd& xds) const
  {
    Eigen::MatrixXd forcing_terms, fa_output;
    analyticalSolution(ts, xs, xds, forcing_terms, fa_output);
  }

  /**
   * Return analytical solution of the system at certain times
   *
   * \param[in]  ts  A vector of times for which to compute the analytical
   * solutions \param[out] trajectory The computed states as a trajectory.
   */
  virtual void analyticalSolution(const Eigen::VectorXd& ts,
                                  Trajectory& trajectory) const
  {
    Eigen::MatrixXd xs, xds;
    analyticalSolution(ts, xs, xds);
    statesAsTrajectory(ts, xs, xds, trajectory);
  }

  /**
   * Return analytical solution of the system at certain times
   *
   * \param[in]  ts  A vector of times for which to compute the analytical
   * solutions \param[out] trajectory The computed states as a trajectory.
   * \param[out] forcing_terms The forcing terms
   */
  inline void analyticalSolution(const Eigen::VectorXd& ts,
                                 Trajectory& trajectory,
                                 Eigen::MatrixXd& forcing_terms) const
  {
    Eigen::MatrixXd xs, xds;
    analyticalSolution(ts, xs, xds, forcing_terms);
    statesAsTrajectory(ts, xs, xds, trajectory);
  }

  /** Get the output of a DMP dynamical system as a trajectory.
   *  As a dynamical system, the state vector of a DMP contains the output of
   * the goal, spring, phase and gating system. What we are most interested in
   * is the output of the spring system. This function extracts that
   * information, and also computes the accelerations of the spring system,
   * which are only stored implicitely in xd_in because second order systems are
   * converted to first order systems with expanded state.
   *
   * \param[in] x_in  State vector over time (size n_time_steps X dim())
   * \param[in] xd_in State vector over time (rates of change)
   * \param[out] y_out  State vector over time (size n_time_steps X dim_y())
   * \param[out] yd_out  State vector over time (rates of change)
   * \param[out] ydd_out  State vector over time (rates of change of rates of
   * change)
   *
   */
  virtual void statesAsTrajectory(const Eigen::MatrixXd& x_in,
                                  const Eigen::MatrixXd& xd_in,
                                  Eigen::MatrixXd& y_out,
                                  Eigen::MatrixXd& yd_out,
                                  Eigen::MatrixXd& ydd_out) const;

  /** Get the output of a DMP dynamical system as a trajectory.
   *  As a dynamical system, the state vector of a DMP contains the output of
   * the goal, spring, phase and gating system. What we are most interested in
   * is the output of the spring system. This function extracts that
   * information, and also computes the accelerations of the spring system,
   * which are only stored implicitely in xd_in because second order systems are
   * converted to first order systems with expanded state.
   *
   * \param[in] ts    A vector of times
   * \param[in] x_in  State vector over time
   * \param[in] xd_in State vector over time (rates of change)
   * \param[out] trajectory Trajectory representation of the DMP state vector
   * output.
   *
   */
  virtual void statesAsTrajectory(const Eigen::VectorXd& ts,
                                  const Eigen::MatrixXd& x_in,
                                  const Eigen::MatrixXd& xd_in,
                                  Trajectory& trajectory) const;

  /**
   * Accessor function for the time constant.
   * \param[in] tau Time constant
   * We need to override DynamicalSystem::set_tau, because the DMP must also
   * change the time constant of all of its subsystems.
   */
  virtual void set_tau(double tau);

  /** Accessor function for the initial state of the system.
   *  \param[in] y_init Initial state of the system.
   * We need to override DynamicalSystem::set_initial_state, because the DMP
   * must also change the initial state  of the goal system as well.
   */
  virtual void set_y_init(const Eigen::VectorXd& y_init);

  /** Accessor function for the attractor state of the system.
   *  \param[in] y_attr Attractor state of the system.
   */
  virtual void set_y_attr(const Eigen::VectorXd& y_attr);

  /**
   * Accessor function for damping coefficient of spring-damper system
   * \param[in] damping_coefficient Damping coefficient
   */
  void set_damping_coefficient(double damping_coefficient);

  /**
   * Accessor function for spring constant of spring-damper system
   * \param[in] spring_constant Spring constant
   */
  void set_spring_constant(double spring_constant);

  /** Get a pointer to the function approximator for a certain dimension.
   * \param[in] i_dim Dimension for which to get the function approximator
   * \return Pointer to the function approximator.
   */
  inline FunctionApproximator* function_approximator(int i_dim) const
  {
    assert(i_dim < (int)function_approximators_.size());
    return function_approximators_[i_dim];
  }

  /**
   * Get the dimensionality of the dynamical system, i.e. the size of its
   * output.
   *
   * 2nd order systems are represented as 1st order systems with an expanded
   * state. The SpringDamperSystem for instance is represented as x = [y z], xd
   * = [yd zd]. DynamicalSystem::dim_ is dim(x) = dim([y z]) = 2*dim(y)
   * Dmp::dim_y() instead is dim(y)
   *
   * For Dynamical Movement Primitives, dim_orig() may be for instance 3, if the
   * output of the DMP represents x,y,z coordinates. However, dim_ will have a
   * much larger dimensionality, because it also contains the variables of all
   * the subsystems (phase system, gating system, etc.)
   *
   * \return Dimensionality of the DMP (number of tranformation systems)
   */
  inline int dim_dmp(void) const { return dim_y(); }

  /**
   * Accessor function for the initial state of the dynamical system.
   * \return y_init Initial state of the dynamical system.
   */
  inline Eigen::VectorXd y_init(void) const
  {
    return x_init().segment(0, dim_y());
  }

  /**
   * Accessor function for the initial state of the dynamical system.
   * \param[out] y_init Initial state of the dynamical system.
   */
  inline void get_y_init(Eigen::VectorXd& y_init) const
  {
    // x = [y z etc], return only y part
    y_init = x_init().segment(0, dim_y());
  }

  /** Mutator function for the initial state of the dynamical system.
   *  \param[in] y_init Initial state of the dynamical system.
   */
  inline void set_y_init(const Eigen::Ref<const Eigen::VectorXd>& y_init)
  {
    set_x_init(y_init);
  }

  /** Read an object from json.
   *  \param[in]  j json input
   *  \param[out] obj The object read from json
   *
   * See also: https://github.com/nlohmann/json/issues/1324
   */
  friend void from_json(const nlohmann::json& j, Dmp*& obj);

  /** Write an object to json.
   *  \param[in] obj The object to write to json
   *  \param[out]  j json output
   *
   * See also:
   *   https://github.com/nlohmann/json/issues/1324
   *   https://github.com/nlohmann/json/issues/716
   */
  inline friend void to_json(nlohmann::json& j, const Dmp* const& obj)
  {
    obj->to_json_helper(j);
  }

 private:
  /** Write this object to json.
   *  \param[out]  j json output
   *
   * See also:
   *   https://github.com/nlohmann/json/issues/1324
   *   https://github.com/nlohmann/json/issues/716
   */
  void to_json_helper(nlohmann::json& j) const;

  /** The attractor state of the system, to which the system will converge. */
  Eigen::VectorXd y_attr_;

  /** @name Linear closed loop controller
   *  @{
   */
  /** Delayed goal system. Also see \ref sec_delayed_goal */
  ExponentialSystem* goal_system_;

  /** Spring-damper system. Also see \ref page_dyn_sys */
  SpringDamperSystem* spring_system_;
  /** @} */  // end of group_linear

  /** @name Non-linear open loop controller
   *  @{
   */
  /** System that determined the phase of the movement. */
  DynamicalSystem* phase_system_;
  /** System to gate the output of the function approximators. Starts at 1 and
   * converges to 0. */
  DynamicalSystem* gating_system_;

  /** The function approximators, one for each dimension, in the forcing term.
   */
  std::vector<FunctionApproximator*> function_approximators_;

  /** How is the forcing term scaled? */
  std::string forcing_term_scaling_;

  /** Ranges of the trajectory (per dimension) for (optional) scaling of forcing
   * term.  */
  Eigen::VectorXd trajectory_amplitudes_;

  /** @} */  // end of group_nonlinear

  /** Pre-allocated memory to avoid allocating run-time (for real-time). */
  mutable Eigen::VectorXd y_init_prealloc_;

  /** Pre-allocated memory to avoid allocating run-time (for real-time). */
  mutable Eigen::VectorXd fa_output_one_prealloc_;

  /** Pre-allocated memory to avoid allocating run-time (for real-time). */
  mutable Eigen::MatrixXd fa_output_prealloc_;

  /** Pre-allocated memory to avoid allocating run-time (for real-time). */
  mutable Eigen::VectorXd forcing_term_prealloc_;

  /** Pre-allocated memory to avoid allocating run-time (for real-time). */
  mutable Eigen::VectorXd g_minus_y0_prealloc_;

  /**
   *  Helper function for constructor.
   *
   * \param[in] spring_system  Spring-damper system cf. Dmp::spring_system_
   * \param[in] goal_system    System to compute delayed goal, cf.
   * Dmp::damping_coefficient_
   * \param[in] phase_system    System to compute the phase, cf.
   * Dmp::phase_system_
   * \param[in] gating_system   System to compute the gating
   * term, cf. Dmp::gating_system_
   */
  void initSubSystems(double alpha_spring_system,
                      ExponentialSystem* goal_system,
                      DynamicalSystem* phase_system,
                      DynamicalSystem* gating_system);

  void initSubSystems(std::string dmp_type);

  /**
   *  Helper function for constructor.
   *
   * \param[in] function_approximators Function
   * approximators for the forcing term, cf. Dmp::function_approximators_
   */
  void initFunctionApproximators(
      std::vector<FunctionApproximator*> function_approximators);
};

}  // namespace DmpBbo

#endif  // _DMP_H_

namespace DmpBbo {

/** \page page_dmp Dynamical Movement Primitives

This page provides an  overview of the implementation of DMPs in the \c dmps/
module.

It is assumed you have read about the theory behind DMPs in the tutorial <a
href="https://github.com/stulp/dmpbbo/blob/master/tutorial/dmp.md">tutorial/dmp.md</a>.
Note that in the tutorial, we have used the notation \f$[z~y]\f$ for consistency
with the DMP literature. In the C++ implementation, the order is rather
\f$[y~z]\f$.


Since a Dynamical Movement Primitive is a dynamical system, the Dmp class
derives from the DynamicalSystem class. It overrides the virtual function
DynamicalSystem::integrateStart(). Integrating the DMP numerically is done
with the generic DynamicalSystem::integrateStep()
function. It also implements the pure virtual function
DynamicalSystem::analyticalSolution(). Because a DMP cannot be solved
analytically (we cannot write it in closed form due to the arbitrary forcing
term), calling Dmp::analyticalSolution() in fact performs a numerical Euler
integration (although the linear subsystems (phase, gating, etc.) are
analytically solved because this is faster computationally).

\em Remark. Dmp inherits the function DynamicalSystem::integrateStep() which
calls DynamicalSystem::integrateStepRungeKutta(), which performs a 4-th order
Runge-Kutta integration. To use faster but less accurate Euler integration,
call DynamicalSystem::integrateStepEuler() explicitly. Euler is faster
because it requires only 1 call to DynamicalSystem::differentialEquation(),
instead of 4 for 4-th order Runge-Kutta integration.

To numerically integrate a dynamical system, one must carefully choose the
integration time dt. Choosing it too low leads to inaccurate integration, and
the numerical integration will diverge from the 'true' solution acquired through
analytical solution. See http://en.wikipedia.org/wiki/Euler%27s_method for
examples. Choosing dt depends entirely on the time-scale (seconds vs. years) and
parameters of the dynamical system (time constant, decay parameters). For DMPs,
which are expected to take between 0.5-10 seconds, dt is usually chosen to be in
the range 0.01-0.001.

The state of a Dmp includes all its subsystems, i.e. the phase_system,
gating_system, goal_system. If the Dmp has size dim_dmp()=3 (e.g. representing an
end-effector position), then the state will have a size of 11, i.e. 2*3 (one 2nd
order spring-damper expanded to two 1st order systems) + 3 (goal sytem) + 1
(phase system) + gating system (1). For convenience, there are several macros to
extract different pars of the state, e.g. x.GOAL expands to x.segment(2 *
dim_dmp() + 0, dim_dmp()), which extracts the state of the goal system.

*/

}
