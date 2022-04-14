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
 * implement are \li differentialEquation() : The differential equation that
 * defines the system \li analyticalSolution() : The analytical solution to the
 * system at given times
 *
 * This class provides accesor/mutator methods for some variables typically
 * found in dynamical systems: \li dim_ The dimensionality of the system, i.e.
 * the number of variables in the state vector. \li tau() The time constant of
 * the system \li x_init() The initial state of the system \li
 * attractor_state() The attractor state, i.e. the state to which the system
 * will converge
 *
 * This class also provides functionality for integrating a dynamical system by
 * repeatidly calling it's differentialEquation, and doing simple Euler or
 * 4th-order Runge-Kutta integration. The related functions are: \li
 * set_integration_method(IntegrationMethod), i.e. EULER or RUNGE_KUTTA \li
 * integrateStep() Integrate the system one time step, using the current
 * integration method \li integrateStart() Start integrating the system (does
 * not require the current state to be passed)
 *
 * \ingroup DynamicalSystems
 */
class DynamicalSystem {
 public:
  /** @name Constructors/Destructor
   *  @{
   */

  /**
   * Initialization constructor.
   * \param order     Order of the system
   * \param tau       Time constant, see tau()
   * \param x_init    Initial state, see x_init()
   */
  DynamicalSystem(int order, double tau, Eigen::VectorXd xy_init);

  /**
   * Initialization constructor.
   * \param tau     Time constant, see tau()
   * \param n_dims  Dimensionality of the state (which may differ from the size
   * of y_init) \param y_init  Part of the initial state Only works for
   * first-order systems (order=1)
   */
  DynamicalSystem(double tau, Eigen::VectorXd y_init, int n_dims);

  /** Destructor */
  virtual ~DynamicalSystem(void);

  /** @} */

  /** @name Main DynamicalSystem functions
   *  @{
   */

  /**
   * The differential equation which defines the system.
   * It relates state values to rates of change of those state values
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
   * Return analytical solution of the system at certain times
   *
   * \param[in]  ts  A vector of times for which to compute the analytical
   * solutions \param[out] xs  Sequence of state vectors. T x D or D x T matrix,
   * where T is the number of times (the length of 'ts'), and D the size of the
   * state (i.e. dim_) \param[out] xds Sequence of state vectors (rates of
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

  /** Start integrating the system with a new initial state
   *
   * \param[in]  x_init          - The initial state vector
   * \param[out] x               - The first vector of state variable
   * \param[out] xd              - The first vector of rates of change of the
   * state variables
   *
   */
  void integrateStart(const Eigen::VectorXd& x_init,
                      Eigen::Ref<Eigen::VectorXd> x,
                      Eigen::Ref<Eigen::VectorXd> xd);

  /**
   * Integrate the system one time step.
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
   * \remarks x should be of size dim_ X 1. This forces you to pre-allocate
   * memory. As a consequence, if differentialEquation is real-time
   * in a derived class, integrateStep will be real-time also.
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
  inline int dim(void) const { return dim_; }

  /**
   * Accessor function for the time constant.
   * \return Time constant
   */
  inline double tau(void) const { return tau_; }

  /**
   * Mutator function for the time constant.
   * \param[in] tau Time constant
   */
  inline virtual void set_tau(double tau) { tau_ = tau; }

  /**
   * Accessor function for the initial state of the dynamical system.
   * \param[out] initial_state Initial state of the dynamical system.
   */
  inline Eigen::VectorXd x_init(void) const { return x_init_; }

  /**
   * Accessor function for the initial state of the dynamical system.
   * \param[out] initial_state Initial state of the dynamical system.
   */
  inline void get_x_init(Eigen::VectorXd& x_init) const { x_init = x_init_; }

  /** Mutator function for the initial state of the dynamical system.
   *  \param[in] initial_state Initial state of the dynamical system.
   */
  virtual void set_x_init(const Eigen::VectorXd& x_init);

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

  /** Dimensionality of the system.
   *  For instance, if there are 3 state variables, dim_ is 3.
   */
  const int dim_;

  /** Time constant */
  double tau_;

  /** The initial state of the system.
   *  It is a column vector of size dim_orig().
   */
  Eigen::VectorXd x_init_;

  void preallocateMemory(int dim);

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

\section sec_dyn_sys_analytical_solution Analytical solution of a dynamical
system

In the object-oriented implementation of this module, all dynamical systems
inherit from the abstract DynamicalSystem class. The analytical solution of a
dynamical system is computed with DynamicalSystem::analyticalSolution, which
takes the times \c ts at which the solution should be computed, and returns the
evolution of the system as \c xs and \c xds.

\section sec_dyn_sys_numeric_integration Numeric integration of a dynamical
system

A system's differential equation is implement in the function
DynamicalSystem::differentialEquation, which takes the current state \c x, and
computes the rates of change \c xd. The functions
DynamicalSystem::integrateStart() and DynamicalSystem::integrateStep() are then
used to numerically integrate the system as follows (using the example plotted
above):

\code
// Make exponential system that decays from 4 to 0 with decay constant 6, and
tau=1.0 double alpha = 6.0;                // Decay constant double tau = 1.0;
// Time constant VectorXd x_init(1); x_init << 4.0; // Initial state (a 1D
vector with the value 4.0 inside using Eigen comma initializer) VectorXd
x_attr(1); x_attr << 0.0; // Attractor state DynamicalSystem* dyn_sys = new
ExponentialSystem(tau, x_init, x_attr, alpha);

Eigen::VectorXd x, xd;
dyn_sys->integrateStart(x,xd); // Start the integration
double dt = 0.01;
for (double t=0.0; t<1.5; t+=dt)
{
  dyn_sys->integrateStep(dt,x,x,xd);          // Takes current state x,
integrates system, and writes next state in x, xd cout << t << " " << x << " "
<< xd << endl; // Output current time, state and rate of change
}
delete dyn_sys;
\endcode

\em Remark. Both analyticalSolution and differentialEquation functions above are
const, i.e. they do not change the DynamicalSystem itself.

\em Remark. DynamicalSystem::integrateStep() uses either Euler integration, or
4-th order Runge-Kutta.  Runge-Kutta is much more accurate, but requires 4 calls
of DynamicalSystem::differentialEquation() instead of only 1 for Euler
integration. Which one is used can be set with
DynamicalSystem::set_integration_method(). To numerically integrate a dynamical
system, one must carefully choose the integration time dt. Choosing it too low
leads to inaccurate integration, and the numerical integration will diverge from
the "true" solution acquired through analytical solution. See
http://en.wikipedia.org/wiki/Euler%27s_method for examples. Choosing dt depends
entirely on the time-scale (seconds vs. years) and parameters of the dynamical
system (time constant, decay parameters).


\section dyn_sys_rewrite_second_first Rewriting one 2nd Order Systems as two 1st
Order Systems

For the theory, behind this see
<a
href="../../../tutorial/dynamicalsystems.md#dyn_sys_second_order_systems">tutorial/dynamicalsystems.md</a>.

The constructor DynamicalSystem::DynamicalSystem() immediately converts second
order systems, such as SpringDamperSystem, into first order systems with an
expanded state.

The function DynamicalSystem::dim_ returns the size of the entire state vector
\f$ \mathbf{x} = [y~z]\f$, the function DynamicalSystem::dim_orig() return the
size of only the \f$ y \f$ component. The attractor and initial stateof the
dynamical system must always have the size returned by
DynamicalSystem::dim_orig().

*/

}
