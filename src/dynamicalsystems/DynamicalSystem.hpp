/**
 * @file DynamicalSystem.hpp
 * @brief  DynamicalSystem class header file.
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

#ifndef _DYNAMICALSYSTEM_H_
#define _DYNAMICALSYSTEM_H_

#define EIGEN_RUNTIME_NO_MALLOC  // Enable runtime tests for allocations

#include <eigen3/Eigen/Core>
#include <nlohmann/json_fwd.hpp>

namespace DmpBbo {

/** \defgroup DynamicalSystems Dynamical Systems Module
 */

/** \brief Interface for implementing dynamical systems. Other dynamical systems
 * should inherit from this class.
 *
 * See also the \ref page_dyn_sys page
 *
 * Two pure virtual functions that each DynamicalSystem subclass should
 * implement are
 * \li differentialEquation() : The differential equation that
 * defines the system
 * \li analyticalSolution() : The analytical solution to the
 * system at given times
 *
 * This class provides accesor/mutator methods for some variables typically
 * found in dynamical systems:
 * \li dim() The dimensionality of the system, i.e.
 * the number of variables in the state vector.
 * \li tau() The time constant of
 * the system
 * \li x_init() The initial state of the system
 * \li attractor_state() The attractor state, i.e. the state to which the system
 * will converge
 *
 * \ingroup DynamicalSystems
 */
class DynamicalSystem {
 public:
  /** @name Constructors/Destructor
   *  @{
   */

  /**
   * Initialize a first or second order dynamical system.
   *
   * \param[in] order     Order of the system (1 or 2)
   * \param[in] tau       Time constant
   * \param[in] y_init    Initial state
   */
  DynamicalSystem(int order, double tau, Eigen::VectorXd y_init);

  /**
   * Initialize a first order dynamical system, with extra state variables.
   *
   * \param[in] tau      Time constant
   * \param[in] y_init   Initial state
   * \param[in] n_dims_x Dimensionality of the state (which may differ from the
   * size of y_init)
   */
  DynamicalSystem(double tau, Eigen::VectorXd y_init, int n_dims_x);

  /** Destructor */
  virtual ~DynamicalSystem(void);

  /** @} */

  /** @name Main DynamicalSystem functions
   *  @{
   */

  /**
   * The differential equation which defines the system.
   * It relates state values to rates of change of those state values.
   *
   * \param[in]  x  current state (column vector of size dim_ X 1)
   * \param[out] xd rate of change in state (column vector of size dim_ X 1)
   *
   * \remarks x and xd should be of size dim_ X 1. This forces you to
   * pre-allocate memory, which speeds things up (and also makes Eigen's Ref
   * functionality easier to deal with).
   */
  virtual void differentialEquation(const Eigen::Ref<const Eigen::VectorXd>& x,
                                    Eigen::Ref<Eigen::VectorXd> xd) const = 0;

  /**
   * Return analytical solution of the system at certain times.
   *
   * \param[in]  ts  A vector of times for which to compute the analytical
   * solutions
   * \param[out] xs  Sequence of state vectors. T x D or D x T matrix,
   * where T is the number of times (the length of 'ts'), and D the size of the
   * state (i.e. dim_)
   * \param[out] xds Sequence of state vectors (rates of
   * change). T x D or D x T matrix, where T is the number of times (the length
   * of 'ts'), and D the size of the state (i.e. dim_)
   *
   * \remarks The output xs and xds will be of size D x T \em only if the matrix
   * x you pass as an argument of size D x T. In all other cases (i.e. including
   * passing an empty matrix) the size of x will be T x D. This feature has been
   * added so that you may pass matrices of either size.
   */
  virtual void analyticalSolution(const Eigen::VectorXd& ts,
                                  Eigen::MatrixXd& xs,
                                  Eigen::MatrixXd& xds) const = 0;

  /** Start integrating the system with a new initial state
   *
   * \param[in]  y_init          - The initial state vector (y part)
   * \param[out] x               - The first vector of state variable
   * \param[out] xd              - The first vector of rates of change of the
   * state variables
   *
   */
  virtual void integrateStart(const Eigen::VectorXd& y_init,
                              Eigen::Ref<Eigen::VectorXd> x,
                              Eigen::Ref<Eigen::VectorXd> xd);

  /** Start integrating the system
   *
   * \param[out] x               - The first vector of state variables
   * \param[out] xd              - The first vector of rates of change of the
   * state variables
   *
   * \remarks x and xd should be of size dim_ X 1. This forces you to
   * pre-allocate memory, which speeds things up (and also makes Eigen's Ref
   * functionality easier to deal with).
   */
  virtual void integrateStart(Eigen::Ref<Eigen::VectorXd> x,
                              Eigen::Ref<Eigen::VectorXd> xd) const;

  /**
   * Integrate the system one time step.
   *
   * \param[in]  dt         Duration of the time step
   * \param[in]  x          Current state
   * \param[out] x_updated  Updated state, dt time later.
   * \param[out] xd_updated Updated rates of change of state, dt time later.
   *
   * \remarks If x_updated and xd_updated are of the correct size (i.e. dim_x),
   * then this function will not allocate memory.
   */
  virtual void integrateStep(double dt,
                             const Eigen::Ref<const Eigen::VectorXd> x,
                             Eigen::Ref<Eigen::VectorXd> x_updated,
                             Eigen::Ref<Eigen::VectorXd> xd_updated) const
  {
    integrateStepRungeKutta(dt, x, x_updated, xd_updated);
  }

  /**
   * Integrate the system one time step using simple Euler integration
   *
   * See http://en.wikipedia.org/wiki/Euler_integration
   *
   * \param[in]  dt         Duration of the time step
   * \param[in]  x          Current state
   * \param[out] x_updated  Updated state, dt time later.
   * \param[out] xd_updated Updated rates of change of state, dt time later.
   *
   * \remarks If x_updated and xd_updated are of the correct size (i.e. dim_x),
   * then this function will not allocate memory.
   */
  void integrateStepEuler(double dt, const Eigen::Ref<const Eigen::VectorXd> x,
                          Eigen::Ref<Eigen::VectorXd> x_updated,
                          Eigen::Ref<Eigen::VectorXd> xd_updated) const;

  /**
   * Integrate the system one time step using 4th order Runge-Kutta integration
   *
   * See
   * http://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Classic_fourth-order_method
   *
   * \param[in]  dt         Duration of the time step
   * \param[in]  x          Current state
   * \param[out] x_updated  Updated state, dt time later.
   * \param[out] xd_updated Updated rates of change of state, dt time later.
   *
   * \remarks x should be of size dim_ X 1. This forces you to pre-allocate
   * memory. As a consequence, if differentialEquation is real-time
   * in a derived class, integrateStep will be real-time also.
   */
  void integrateStepRungeKutta(double dt,
                               const Eigen::Ref<const Eigen::VectorXd> x,
                               Eigen::Ref<Eigen::VectorXd> x_updated,
                               Eigen::Ref<Eigen::VectorXd> xd_updated) const;
  /** @} */

  /** @name Input/Output
   *  @{
   */

  /** Write a DynamicalSystem to an output stream.
   *
   *  \param[in] output  Output stream to which to write to
   *  \param[in] d Dynamical system to write
   *  \return    Output stream
   */
  friend std::ostream& operator<<(std::ostream& output,
                                  const DynamicalSystem& d);

  /** @} */

  /** @name Accessor functions
   *  @{
   */

  /**
   * Get the dimensionality of the dynamical system, i.e. the length of its
   * state vector. \return Dimensionality of the dynamical system
   */
  inline int dim(void) const { return dim_x_; }
  inline int dim_x(void) const { return dim_x_; }
  inline int dim_y(void) const { return dim_y_; }

  /**
   * Get the time constant.
   * \return Time constant
   */
  inline double tau(void) const { return tau_; }

  /**
   * Set the time constant.
   * \param[in] tau Time constant
   */
  inline virtual void set_tau(double tau) { tau_ = tau; }

  /**
   * Get the initial state of the dynamical system.
   * \return Initial state of the dynamical system.
   */
  inline Eigen::VectorXd x_init(void) const { return x_init_; }

  /**
   * Get the initial state of the dynamical system.
   * \param[out] x_init Initial state of the dynamical system.
   */
  inline void get_x_init(Eigen::VectorXd& x_init) const { x_init = x_init_; }

  /** Set the initial state of the dynamical system.
   *  \param[in] x_init Initial state of the dynamical system.
   */
  virtual void set_x_init(const Eigen::VectorXd& x_init)
  {
    assert(x_init.size() == dim_x_);
    x_init_ = x_init;
  }

  /**
   * Get the y part of the initial state of the dynamical system.
   *
   * \param[out] y_init Initial state of the dynamical system.
   */
  void get_y_init(Eigen::VectorXd& y_init) const;

  /** Set the y part of the initial state of the dynamical system.
   *  \param[in] y_init Initial state of the dynamical system.
   */
  virtual void set_y_init(const Eigen::Ref<const Eigen::VectorXd>& y_init);

  /** @} */

  /** Read an object from json.
   *  \param[in]  j   json input
   *  \param[out] obj The object read from json
   *
   * See also: https://github.com/nlohmann/json/issues/1324
   */
  friend void from_json(const nlohmann::json& j, DynamicalSystem*& obj);

  /** Write an object to json.
   *  \param[in] obj The object to write to json
   *  \param[out]  j json output
   *
   * See also:
   *   https://github.com/nlohmann/json/issues/1324
   *   https://github.com/nlohmann/json/issues/716
   */
  inline friend void to_json(nlohmann::json& j,
                             const DynamicalSystem* const& obj)
  {
    obj->to_json_helper(j);
  }

 protected:
  /** Write an members from this base class to to json.
   *  \param[out]  j json output
   *
   * See also: https://github.com/nlohmann/json/issues/716
   */
  void to_json_base(nlohmann::json& j) const;

 private:
  /** Write this object to json.
   *  \param[out]  j json output
   *
   * See also:
   *   https://github.com/nlohmann/json/issues/1324
   *   https://github.com/nlohmann/json/issues/716
   */
  virtual void to_json_helper(nlohmann::json& j) const = 0;

  /** Dimensionality of the system state.
   *
   *  For instance, if there are 3 state variables, dim_ is 3.
   *
   * For second order systems, it is dim(x) = dim([y z])
   */
  const int dim_x_;

  /** Dimensionality of the y part of the system state.
   *
   * For first order systems, it is dim(y) = dim(x) (because y = x)
   *
   * For second order systems, it is dim(y), where y is part of x, i.e. x = [y
   * z]
   */
  const int dim_y_;

  /** Time constant */
  double tau_;

  /** The initial state of the system.
   *  It is a column vector of size dim_orig().
   */
  Eigen::VectorXd x_init_;

  void preallocateMemory(void);

  /** Members for caching in Runge-Kutta integration. */
  mutable Eigen::VectorXd k1_, k2_, k3_, k4_, input_k2_, input_k3_, input_k4_;
};

}  // namespace DmpBbo

#endif  // _DYNAMICALSYSTEM_H_

namespace DmpBbo {

/** \page page_dyn_sys Dynamical Systems

This page provides an  overview of the implementation of dynamical systems in
the \c dynamicalsystems/ module.

It is assumed you have read about the theory behind dynamical systems in the
tutorial <a
href="https://github.com/stulp/dmpbbo/blob/master/tutorial/dynamicalsystems.md">tutorial/dynamicalsystems.md</a>.

\section sec_dyn_sys_ana Analytical solution of a dynamical system

In the object-oriented implementation of this module, all dynamical systems
inherit from the abstract class DynamicalSystem. The analytical solution of a
dynamical system is computed with DynamicalSystem::analyticalSolution, which
takes the times \c ts at which the solution should be computed, and returns the
evolution of the system as \c xs and \c xds (of size: n_time_steps X n_dim)

\section sec_dyn_sys_numeric_integration Numeric integration of a dynamical
system

A system's differential equation is implement in the function
DynamicalSystem::differentialEquation, which takes the current state \c x, and
computes the rates of change \c xd. The functions
DynamicalSystem::integrateStart() and DynamicalSystem::integrateStep() are then
used to numerically integrate the system as follows (using the example plotted
above):

\code
// Make exponential system that decays from 4 to 0 with decay constant 6

double alpha = 6.0; // Decay constant
double tau = 1.0; // Time constant
VectorXd x_init(1); x_init << 4.0; // Initial state
VectorXd x_attr(1); x_attr << 0.0; // Attractor state

DynamicalSystem* dyn_sys = new ExponentialSystem(tau, x_init, x_attr, alpha);

Eigen::VectorXd x, xd;
dyn_sys->integrateStart(x,xd); // Start the integration
double dt = 0.01;
for (double t=0.0; t<1.5; t+=dt) {
  dyn_sys->integrateStep(dt,x,x,xd);
  // Takes current state x, integrates system, and writes next state in x, xd
  cout << t << " " << x << " " << xd << endl;
}
delete dyn_sys;
\endcode

\em Remark. Both analyticalSolution and differentialEquation functions above are
const, i.e. they do not change the DynamicalSystem itself. The state of the
dynamical system is not stored as a member (except for the initial state).

\em Remark. DynamicalSystem::integrateStep() uses either Euler integration
(DynamicalSystem::integrateStepEuler()), or 4-th order Runge-Kutta
(DynamicalSystem::integrateStepRungeKutta()).  Runge-Kutta is much more
accurate, but requires 4 calls of DynamicalSystem::differentialEquation()
instead of only 1 for Euler integration.

\em \remark To numerically integrate a dynamical
system, one must carefully choose the integration time dt. Choosing it too low
leads to inaccurate integration, and the numerical integration will diverge from
the "true" solution acquired through analytical solution. See
http://en.wikipedia.org/wiki/Euler%27s_method for examples. Choosing dt depends
entirely on the time-scale (seconds vs. years) and parameters of the dynamical
system (time constant, decay parameters).


\section dyn_sys_second Rewriting one 2nd Order Systems as two 1st Order Systems

For the theory behind this see
<a
href="../../../tutorial/dynamicalsystems.md#dyn_sys_second_order_systems">tutorial/dynamicalsystems.md</a>.

The constructor DynamicalSystem::DynamicalSystem() immediately converts second
order systems, such as SpringDamperSystem, into first order systems with an
expanded state. The function DynamicalSystem::dim() returns the size of the
entire state vector dim(x), which for second order systems, is equivalent to
dim(\f$ [y z]\f$).

The function DynamicalSystem::x_init() returns the entire state vector. Second
order systems have an additional function for accessing only the y part of the
state SpringDamperSystem::y_init(). For first order system, y is equivalent to
x.

There is a special constructor
DynamicalSystem::DynamicalSystem(double tau, Eigen::VectorXd y_init, int
n_dims); which is necessary for dynamical movement primitives. This constructs a
first order system (which is how DMPs are usually formalized), with the inital
state y_init. The argument n_dims specifies that room should be made (beyone y)
for the other systems (gating, phase, etc.)

*/

}
