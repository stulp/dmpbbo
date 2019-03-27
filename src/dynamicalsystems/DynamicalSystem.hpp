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

#include "eigen_realtime/eigen_realtime_check.hpp" // Include this before Eigen header files

#include <string>
#include <vector>
#include <eigen3/Eigen/Core>

#include "dmpbbo_io/EigenBoostSerialization.hpp"

namespace DmpBbo {

/** \defgroup DynamicalSystems Dynamical Systems Module
 */

/** \brief Interface for implementing dynamical systems. Other dynamical systems should inherit from this class.
 *
 * See also the \ref page_dyn_sys page
 *
 * Two pure virtual functions that each DynamicalSystem subclass should implement are
 * \li differentialEquation() : The differential equation that defines the system
 * \li analyticalSolution() : The analytical solution to the system at given times
 *
 * This class provides accesor/mutator methods for some variables typically found in
 * dynamical systems:
 * \li dim() The dimensionality of the system, i.e. the number of variables in the state vector.
 * \li tau() The time constant of the system
 * \li initial_state() The initial state of the system
 * \li attractor_state() The attractor state, i.e. the state to which the system will converge
 *
 * This class also provides functionality for integrating a dynamical system by repeatidly
 * calling it's differentialEquation, and doing simple Euler or 4th-order Runge-Kutta integration.
 * The related functions are:
 * \li set_integration_method(IntegrationMethod), i.e. EULER or RUNGE_KUTTA
 * \li integrateStep() Integrate the system one time step, using the current integration method
 * \li integrateStart() Start integrating the system (does not require the current state to be passed)
 *
 * \ingroup DynamicalSystems
 */
class DynamicalSystem
{

public:
  
  /** @name Constructors/Destructor
   *  @{
   */ 

  /**
   * Initialization constructor.
   * \param order            Order of the system
   * \param tau              Time constant, see tau()
   * \param initial_state    Initial state, see initial_state()
   * \param attractor_state  Attractor state, see attractor_state()
   * \param name             A name you give, see name()
   */
   DynamicalSystem(int order, double tau, Eigen::VectorXd initial_state, Eigen::VectorXd attractor_state, std::string name);

  /** Destructor */
  virtual ~DynamicalSystem(void);

  /** Return a pointer to a deep copy of the DynamicalSystem object.
   *  \return Pointer to a deep copy
   */
  virtual DynamicalSystem* clone(void) const = 0;
  
  /** @} */ 
   
  /** @name Main DynamicalSystem functions
   *  @{
   */
   
  /**
   * The differential equation which defines the system.
   * It relates state values to rates of change of those state values
   *
   * \param[in]  x  current state (column vector of size dim() X 1)
   * \param[out] xd rate of change in state (column vector of size dim() X 1)
   *
   * \remarks x and xd should be of size dim() X 1. This forces you to pre-allocate memory, which
   * speeds things up (and also makes Eigen's Ref functionality easier to deal with).
   */
   virtual void differentialEquation(
     const Eigen::Ref<const Eigen::VectorXd>& x, 
     Eigen::Ref<Eigen::VectorXd> xd
   ) const = 0;


  /**
   * Return analytical solution of the system at certain times
   *
   * \param[in]  ts  A vector of times for which to compute the analytical solutions
   * \param[out] xs  Sequence of state vectors. T x D or D x T matrix, where T is the number of times (the length of 'ts'), and D the size of the state (i.e. dim())
   * \param[out] xds Sequence of state vectors (rates of change). T x D or D x T matrix, where T is the number of times (the length of 'ts'), and D the size of the state (i.e. dim())
   *
   * \remarks The output xs and xds will be of size D x T \em only if the matrix x you pass as an argument of size D x T. In all other cases (i.e. including passing an empty matrix) the size of x will be T x D. This feature has been added so that you may pass matrices of either size. 
   */
  virtual void analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs, Eigen::MatrixXd& xds) const  = 0;

  /** Start integrating the system
   *
   * \param[out] x               - The first vector of state variables
   * \param[out] xd              - The first vector of rates of change of the state variables
   *
   * \remarks x and xd should be of size dim() X 1. This forces you to pre-allocate memory, which
   * speeds things up (and also makes Eigen's Ref functionality easier to deal with).
   */
  virtual void integrateStart(Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> xd) const;

  /** Start integrating the system with a new initial state
   *
   * \param[in]  x_init          - The initial state vector
   * \param[out] x               - The first vector of state variable
   * \param[out] xd              - The first vector of rates of change of the state variables
   *
   */
  void integrateStart(const Eigen::VectorXd& x_init, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> xd);

  /**
   * Integrate the system one time step.
   *
   * \param[in]  dt         Duration of the time step
   * \param[in]  x          Current state
   * \param[out] x_updated  Updated state, dt time later.
   * \param[out] xd_updated Updated rates of change of state, dt time later.
   *
   * \remarks x should be of size dim() X 1. This forces you to pre-allocate memory, which
   * speeds things up (and also makes Eigen's Ref functionality easier to deal with).
   */
  virtual void integrateStep(double dt, const Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> x_updated, Eigen::Ref<Eigen::VectorXd> xd_updated) const;

  /** @} */ 
  
  /** @name Input/Output
   *  @{
   */ 

  /** Write a DynamicalSystem to an output stream.
   *
   *  \param[in] output  Output stream to which to write to
   *  \param[in] dyn_sys Dynamical system to write
   *  \return    Output stream
   *
   *  \remarks Calls pure virtual function toString(), which must be implemented by
   *  all subclasses: http://stackoverflow.com/questions/4571611/virtual-operator
   */
  friend std::ostream& operator<<(std::ostream& output, const DynamicalSystem& dyn_sys) {
    output << dyn_sys.toString();
    return output;
  }

  /** Returns a string representation of the object.
   * \return A string representation of the object.
   */
  virtual std::string toString(void) const = 0;

  /** @} */ 
  
  /** @name Accessor functions
   *  @{
   */


  /** The possible integration methods that can be used.
   * \li EULER: simple Euler method (http://en.wikipedia.org/wiki/Euler_integration)
   * \li RUNGE_KUTTA: 4th-order Runge-Kutta (http://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Classic_fourth-order_method)
   */
  enum IntegrationMethod { EULER, RUNGE_KUTTA };

  /** Choose the integration method.
   *  \param[in] integration_method The integration method, see DynamicalSystem::IntegrationMethod.
   */
  inline void set_integration_method(IntegrationMethod integration_method) {
    integration_method_ = integration_method;
  }

  /**
   * Get the dimensionality of the dynamical system, i.e. the length of its state vector.
   * \return Dimensionality of the dynamical system
   */
  inline int dim(void) const {
    return dim_;
  }

  /**
   * Get the dimensionality of the dynamical system, i.e. the length of its output.
   *
   * 2nd order systems are represented as 1st order systems with an expanded state. The
   * SpringDamperSystem for instance is represented as x = [y z], xd = [yd zd].
   * DynamicalSystem::dim() returns dim(x) = dim([y z]) = 2*dim(y)
   * DynamicalSystem::dim_orig() returns dim(y) = dim()/2
   *
   * For Dynamical Movement Primitives, dim_orig() may be for instance 3, if the output of the DMP
   * represents x,y,z coordinates. However, dim() will have a much larger dimensionality, because it 
   * also contains the variables of all the subsystems (phase system, gating system, etc.)
   * 
   * \return Original dimensionality of the dynamical system
   */
   inline int dim_orig(void) const {
    return dim_orig_;
   }

  /**
   * Accessor function for the time constant.
   * \return Time constant
   */
  inline double tau(void) const { return tau_; }

  /**
   * Mutator function for the time constant.
   * \param[in] tau Time constant
   */
  inline virtual void set_tau(double tau) {
    assert(tau>0.0);
    tau_ = tau;
  }

  /**
   * Accessor function for the initial state of the dynamical system.
   * \return Initial state of the dynamical system.
   */
  inline Eigen::VectorXd initial_state(void) const { return initial_state_; }

  /**
   * Accessor function for the initial state of the dynamical system.
   * \param[out] initial_state Initial state of the dynamical system.
   */
  inline void initial_state(Eigen::VectorXd& initial_state) const 
  { 
    initial_state=initial_state_;
  }
  
  /** Mutator function for the initial state of the dynamical system.
   *  \param[in] initial_state Initial state of the dynamical system.
   */
  inline virtual void set_initial_state(const Eigen::VectorXd& initial_state) {
    assert(initial_state.size()==dim_orig_);
    initial_state_ = initial_state;
  }

  /**
   * Accessor function for the attractor state of the dynamical system.
   * \return Attractor state of the dynamical system.
   */
  inline Eigen::VectorXd attractor_state(void) const { return attractor_state_; }

  /**
   * Accessor function for the attractor state of the dynamical system.
   * \param[out] attractor_state Attractor state of the dynamical system.
   */
  inline void attractor_state(Eigen::VectorXd& attractor_state) const 
  { 
    attractor_state=attractor_state_;
  }

  /** Mutator function for the attractor state of the dynamical system.
   *  \param[in] attractor_state Attractor state of the dynamical system.
   */
  inline virtual void set_attractor_state(const Eigen::Ref<const Eigen::VectorXd>& attractor_state) {
    assert(attractor_state.size()==dim_orig_);
    attractor_state_ = attractor_state;
  }

  /**
   * Accessor function for the name of the dynamical system.
   * \return Name of the dynamical system.
   */
  inline std::string name(void) const { return name_; }

  /** Mutator function for the name of the dynamical system.
   *  \param[in] name Name of the dynamical system.
   */
  inline virtual void set_name(std::string name) {
    name_ = name;
  }
  
protected:
  /**
   * Set the dimensionality of the dynamical system, i.e. the length of its state vector.
   * \param[in] dim Dimensionality of the dynamical system
   */
  inline void set_dim(int dim) {
    dim_ = dim;
  }

  /** @} */
  
private:

  /**
   * Integrate the system one time step using simple Euler integration
   *
   * \param[in]  dt         Duration of the time step
   * \param[in]  x          Current state
   * \param[out] x_updated  Updated state, dt time later.
   * \param[out] xd_updated Updated rates of change of state, dt time later.
   */
  void integrateStepEuler(double dt, const Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> x_updated, Eigen::Ref<Eigen::VectorXd> xd_updated) const;

  /**
   * Integrate the system one time step using 4th order Runge-Kutta integration
   *
   * \param[in]  dt         Duration of the time step
   * \param[in]  x          Current state
   * \param[out] x_updated  Updated state, dt time later.
   * \param[out] xd_updated Updated rates of change of state, dt time later.
   */
  void integrateStepRungeKutta(double dt, const Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> x_updated, Eigen::Ref<Eigen::VectorXd> xd_updated) const;

  /** Dimensionality of the system.
   *  For instance, if there are 3 state variables, dim_ is 3.
   */
  int dim_;

 /** The original dimensionality of the system, see DynamicalSystem::dim_orig()
  */
 int dim_orig_;

  /** Time constant
   *  \remarks The reason that tau_ is protected and not private is that it is usually required by
   *  DynamicalSystem::differentialEquation, which should be as fast as possible (i.e. to avoid
   *  function calls).
   */
  double tau_;

  /** The initial state of the system.
   *  It is a column vector of size dim_orig().
   */
  Eigen::VectorXd initial_state_;

  /** The attractor state of the system, to which the system will converge.
   *  It is a column vector of size dim_orig()
   *  \remarks The reason that attractor_state_ is protected and not private is that it is usually
   *  required by DynamicalSystem::differentialEquation, which should be as fast as possible (i.e.
   *  to avoid function calls and copying of vectors).
   */
  Eigen::VectorXd attractor_state_;

  /** A name you may give to the system.
   * Has no direct functionality, but may perhaps be useful for output or debugging purposes.
   */
  std::string name_;

  /** Which integration method to use. See DynamicalSystem::IntegrationMethod */
  IntegrationMethod integration_method_;
  

protected:
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. See \ref sec_boost_serialization_ugliness
   */
  DynamicalSystem(void) {};
   
private:
  /** Give boost serialization access to private members. */  
  friend class boost::serialization::access;
  
  /** Serialize class data members to boost archive. 
   * \param[in] ar Boost archive
   * \param[in] version Version of the class
   * See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/tutorial.html#simplecase
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & BOOST_SERIALIZATION_NVP(dim_);
    ar & BOOST_SERIALIZATION_NVP(dim_orig_);
    // Const doesn't work, see sec_boost_serialization_ugliness in the docu
    //ar & boost::serialization::make_nvp("dim_orig_", const_cast<int&>(dim_orig_));
    ar & BOOST_SERIALIZATION_NVP(tau_);
    ar & BOOST_SERIALIZATION_NVP(initial_state_);
    ar & BOOST_SERIALIZATION_NVP(attractor_state_);
    ar & BOOST_SERIALIZATION_NVP(name_);
    ar & BOOST_SERIALIZATION_NVP(integration_method_);
  }

};

}

#include <boost/serialization/assume_abstract.hpp>
/** Don't add version information to archives. */
BOOST_SERIALIZATION_ASSUME_ABSTRACT(DmpBbo::DynamicalSystem);
 
#include <boost/serialization/export.hpp>
/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::DynamicalSystem,boost::serialization::object_serializable);

#endif // _DYNAMICALSYSTEM_H_

namespace DmpBbo {

/** \page page_dyn_sys Dynamical Systems

This page provides and overview of the implementation of dynamical systems in the \c dynamicalsystems/ module.

It is assumed you have read about the theory behind dynamical systems in the tutorial <a href="https://github.com/stulp/dmpbbo/tutorial/dynamicalsystems.md">tutorial/dynamicalsystems.md</a>.

\section sec_dyn_sys_analytical_solution Analytical solution of a dynamical system

In the object-oriented implementation of this module, all dynamical systems inherit from the abstract DynamicalSystem class. The analytical solution of a dynamical system is computed with DynamicalSystem::analyticalSolution, which takes the times \c ts at which the solution should be computed, and returns the evolution of the system as \c xs and \c xds.

\section sec_dyn_sys_numeric_integration Numeric integration of a dynamical system

A system's differential equation is implement in the function DynamicalSystem::differentialEquation, which takes the current state \c x, and computes the rates of change \c xd. The functions DynamicalSystem::integrateStart() and DynamicalSystem::integrateStep() are then used to numerically integrate the system as follows (using the example plotted above):

\code
// Make exponential system that decays from 4 to 0 with decay constant 6, and tau=1.0
double alpha = 6.0;                // Decay constant
double tau = 1.0;                  // Time constant 
VectorXd x_init(1); x_init << 4.0; // Initial state (a 1D vector with the value 4.0 inside using Eigen comma initializer)
VectorXd x_attr(1); x_attr << 0.0; // Attractor state
DynamicalSystem* dyn_sys = new ExponentialSystem(tau, x_init, x_attr, alpha);

Eigen::VectorXd x, xd;
dyn_sys->integrateStart(x,xd); // Start the integration
double dt = 0.01;
for (double t=0.0; t<1.5; t+=dt)
{
  dyn_sys->integrateStep(dt,x,x,xd);          // Takes current state x, integrates system, and writes next state in x, xd
  cout << t << " " << x << " " << xd << endl; // Output current time, state and rate of change
}
delete dyn_sys;
\endcode

\em Remark. Both analyticalSolution and differentialEquation functions above are const, i.e. they do not change the DynamicalSystem itself.

\em Remark. DynamicalSystem::integrateStep() uses either Euler integration, or 4-th order Runge-Kutta.  Runge-Kutta is much more accurate, but requires 4 calls of DynamicalSystem::differentialEquation() instead of only 1 for Euler integration. Which one is used can be set with DynamicalSystem::set_integration_method(). To numerically integrate a dynamical system, one must carefully choose the integration time dt. Choosing it too low leads to inaccurate integration, and the numerical integration will diverge from the "true" solution acquired through analytical solution. See http://en.wikipedia.org/wiki/Euler%27s_method for examples. Choosing dt depends entirely on the time-scale (seconds vs. years) and parameters of the dynamical system (time constant, decay parameters). 


\section dyn_sys_rewrite_second_first Rewriting one 2nd Order Systems as two 1st Order Systems

For the theory, behind this see 
<a href="../../../tutorial/dynamicalsystems.md#dyn_sys_second_order_systems">tutorial/dynamicalsystems.md</a>.

The constructor DynamicalSystem::DynamicalSystem() immediately converts second order systems, such as SpringDamperSystem, into first order systems with an expanded state.

The function DynamicalSystem::dim() returns the size of the entire state vector \f$ \mathbf{x} = [y~z]\f$, the function DynamicalSystem::dim_orig() return the size of only the \f$ y \f$ component. The attractor and initial stateof the dynamical system must always have the size returned by DynamicalSystem::dim_orig().

*/

}
