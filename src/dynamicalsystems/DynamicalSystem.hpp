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

#include <string>
#include <vector>
#include <eigen3/Eigen/Core>

#include "dmpbbo_io/EigenBoostSerialization.hpp"

namespace DmpBbo {

/** \defgroup DynamicalSystems Dynamical Systems
 */

/** \brief Interface for implementing dynamical systems. Other dynamical systems should inherit from this class.
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
   virtual void differentialEquation(const Eigen::VectorXd& x, Eigen::Ref<Eigen::VectorXd> xd) const = 0;


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
   * DynamicalSystem::getDim returns dim(x) = dim([y z]) = 2*dim(y)
   * DynamicalSystem::dim_orig returns dim(y) = dim()/2
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

  /** Mutator function for the attractor state of the dynamical system.
   *  \param[in] attractor_state Attractor state of the dynamical system.
   */
  inline virtual void set_attractor_state(const Eigen::VectorXd& attractor_state) {
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



/** \page page_dyn_sys Dynamical Systems Module

\section sec_dyn_sys_intro Introduction

Let a \em state be a vector of real numbers. A dynamical system consists of such a state and a rule that describes how this state will change over time; it describes what future state follows from the current state. A typical example is radioactive decay, where the state \f$x\f$ is the number of atoms, and the rate of decay is \f$\frac{dx}{dt}\f$  proportional to \f$x\f$: \f$ \frac{dx}{dt} = -\alpha x\f$. Here, \f$\alpha\f$ is the `decay constant' and \f$\dot{x}\f$ is a shorthand for \f$\frac{dx}{dt}\f$. Such an evolution rule describes an implicit relation between the current state \f$ x(t) \f$ and the state a short time in the future \f$x(t+dt)\f$.

If we know the initial state of a dynamical system, e.g. \f$x_0\equiv x(0)=4\f$, we may compute the evolution of the state over time through \em numerical \em integration. This means we take the initial state \f$ x_0\f$, and iteratively compute subsequent states \f$x(t+dt)\f$ by computing the rate of change \f$\dot{x}\f$, and integrating this over the small time interval \f$dt\f$. A pseudo-code example is shown below for  \f$x_0\equiv x(0)=4\f$, \f$dt=0.01s\f$ and  \f$\alpha=6\f$.

\code
alpha=6; // Decay constant
dt=0.01; // Duration of one integration step
x=4.0;   // Initial state
t=0.0;   // Initial time
while (t<1.5) {
  dx = -alpha*x;  // Dynamical system rule
  x = x + dx*dt;  // Project x into the future for a small time step dt (Euler integration)
  t = t + dt;     // The future is now!
}
\endcode

This procedure is called ``integrating the system'', and leads the trajectory plotted below (shown for both \f$\alpha=6\f$ and \f$\alpha=3\f$.

\image html exponential_decay-svg.png "Evolution of the exponential dynamical system."
\image latex exponential_decay-svg.pdf "Evolution of the exponential dynamical system." height=4cm

The evolution of many dynamical systems can also be determined analytically, by explicitly solving the differential equation. For instance, \f$N(t) = x_0e^{-\alpha t}\f$ is the solution
to \f$\dot{x} = -\alpha x\f$. Why? Let's plug \f$x(t) = x_0e^{-\alpha t}\f$ into \f$\frac{dx}{dt} = -\alpha x\f$, which leads to \f$\frac{d}{dt}(x_0e^{-\alpha t}) = -\alpha (x_0e^{-\alpha t})\f$. Then derive the left side of the equations, which yields \f$-\alpha(x_0e^{-\alpha t}) = -\alpha (x_0e^{-\alpha t})\f$. QED. Note that the solution works for arbitrary \f$x_0\f$. It should, because the solution should not depend on the initial state.

\subsection dynsys_implementation1 Implementation

<em>
In the object-oriented implementation of this module, all dynamical systems inherit from the abstract DynamicalSystem class. The analytical solution of a dynamical system is computed with DynamicalSystem::analyticalSolution, which takes the times \c ts at which the solution should be computed, and returns the evolution of the system as \c xs and \c xds.

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

\em Remark. DynamicalSystem::integrateStep uses either Euler integration, or 4-th order Runge-Kutta.  The latter is more accurate, but requires 4 calls of DynamicalSystem::differentialEquation() instead of 1). Which one is used can be set with DynamicalSystem::set_integration_method(). To numerically integrate a dynamical system, one must carefully choose the integration time dt. Choosing it too low leads to inaccurate integration, and the numerical integration will diverge from the 'true' solution acquired through analytical solution. See http://en.wikipedia.org/wiki/Euler%27s_method for examples. Choosing dt depends entirely on the time-scale (seconds vs. years) and parameters of the dynamical system (time constant, decay parameters). 

\subsection dynsys_implementation1_plotting Plotting

If you save the output of a dynamical in a file with format (where D is the dimensionality of the system, and T is the number of time steps)
\verbatim
x^0_0 x^1_0 .. x^D_0   xd^0_0 xd^1_0 .. xd^D_0   t_0   
x^0_1 x^1_1 .. x^D_1   xd^0_1 xd^1_1 .. xd^D_1   t_1   
   :     :       :         :      :       :       :    
x^0_T x^1_T .. x^D_T   xd^0_T xd^1_T .. xd^D_T   t_T   
\endverbatim
you can plot this output with 
\code
python dynamicalsystems/plotting/plotDynamicalSystem.py file.txt
\endcode

\subsection dynsys_implementation1_demo Demos

A demonstration of how to initialize and integrate an ExponentialSystem is in demoExponentialSystem.cpp

A more complete demonstration including all implemented dynamical systems is in demoDynamicalSystems.cpp. If you call the resulting executable with a directory argument, e.g.
\code
./demoDynamicalSystems /tmp/demoDynamicalSystems
\endcode
it will save results to file, which you can plot with for instance:
\code
python plotDynamicalSystem.py /tmp/demoDynamicalSystems/ExponentialSystem/results_rungekutta.txt
python plotDynamicalSystem.py /tmp/demoDynamicalSystems/ExponentialSystem/results_euler.txt
\endcode
Different test can be performed with the dynamical system. The test can be chosen by passing further argument, e.g. 
\code
./demoDynamicalSystems /tmp/demoDynamicalSystems rungekutta euler
\endcode
will integrate the dynamical systems with both the Runge-Kutta and simple Euler method. The available tests are:
\li "rungekutta" - Use 4th-order Runge-Kutta integration (more accurate, but more calls of DynamicalSystem::differentialEquation)
\li "euler"      - Use simple Euler integration (less accurate, but faster)
\li "analytical" - Use the analytical solution instead of numerical integration
\li "tau"        - Change tau before doing numerical integration
\li "attractor"  - Change the attractor state during numerical integration
\li "perturb"    - Perturb the state during numerical integration

To compare for instance the analytical solution with the Runge-Kutta integration in a plot, you can do
\code
python plotDynamicalSystemComparison.py /tmp/demoDynamicalSystems/ExponentialSystem/results_analytical.txt  /tmp/demoDynamicalSystems/ExponentialSystem/results_rungekutta.txt
\endcode


</em>

\section sec_dyn_sys_properties Properties and Features of Linear Dynamical Systems

\subsection sec_dyn_sys_convergence Convergence towards the Attractor

In the limit of time, the dynamical system for exponential decay will converge to 0 (i.e. \f$x(\infty) = x_0e^{-\alpha\infty}  = 0\f$). The value 0 is known as the \em attractor of the system.
For simple dynamical systems, it is possible to \em proove that they will converge towards the attractor.

Suppose that the attractor state in our running example is not 0, but 1. In that case, we change the attractor state of the exponential decay to \f$x^g\f$ (\f$g\f$=goal) and define the following differential equation:
\f{eqnarray*}{
\dot{x}  =& -\alpha(x-x^g)                & \mbox{~with attractor } x^g
\f}

This system will now converge to the attractor state \f$x^g\f$, rather than 0.

\image html change_tau_attr-svg.png "Changing the attractor state or time constant."
\image latex change_tau_attr-svg.pdf "Changing the attractor state or time constant." height=4cm

\subsection sec_dyn_sys_perturbations Robustness to Perturbations

Another nice feature of dynamical systems is their robustness to perturbations, which means that they will converge towards the attractor even if they are perturbed. The figure below shows how the perturbed system (cyan) converges towards the attractor state just as the unperturbed system (blue) does.

\image html perturb-svg.png "Perturbing the dynamical system."
\image latex perturb-svg.pdf "Perturbing the dynamical system." height=4cm


\subsection sec_dyn_sys_time_constant Changing the speed of convergence: The time constant

The rates of change computed by the differential equation can be increased or decreased (leading to a faster or slower convergence) with a \em time \em constant, which is usually written as follows:

\f{eqnarray*}{
\tau\dot{x}  =& -\alpha(x-x^g)\\
\dot{x}  =& (-\alpha(x-x^g))/\tau
\f}

\em Remark. For an exponential system, decreasing the time constant \f$\tau\f$ has the same effect as increasing  \f$\alpha\f$. For more complex dynamical systems with several parameters, it is useful to have a separate parameter that changes only the speed of convergence, whilst leaving the other parameters the same.

\subsection sec_dyn_sys_multi Multi-dimensional states

The state \f$x\f$ need not be a scalar, but may be a vector. This then represents a multi-dimensional state, i.e. \f$\tau\dot{\mathbf{x}}  = -\alpha(\mathbf{x}-\mathbf{x}^g)\f$. In the code, the size of the state vector \f$dim(\mathbf{x})\equiv dim(\dot{\mathbf{x}})\f$ of a dynamical system is returned by the function DynamicalSystem::dim()

\subsection sec_dyn_sys_autonomy Autonomy

Dynamical system that do not depend on time are called \em autonomous. For instance, the formula \f$ \dot{x}  = -\alpha x\f$ does not depend on time, which means the exponential system is autonomous.


\subsection Implementation
<em>
The attractor state and time constant of a dynamical system are usually passed to the constructor. They can be changed afterwards with with DynamicalSystem::set_attractor_state and DynamicalSystem::set_tau. Before integration starts, the initial state can be set with  DynamicalSystem::set_initial_state. This influences the output of DynamicalSystem::integrateStart, but not DynamicalSystem::integrateStep.

Further (first order) linear dynamical systems that are implemented in this module is a SigmoidSystem (see
 http://en.wikipedia.org/wiki/Exponential_decay and http://en.wikipedia.org/wiki/Sigmoid_function), as well as a dynamical system that has a constant velocity (TimeSystem), so as to mimic the passing of time (time moves at a constant rate per time ;-)
 
\f{eqnarray*}{
\dot{x}  =& -\alpha (x-x^g)    & \mbox{exponential decay/growth} \label{equ_}\\
\dot{x}  =& \alpha x (\beta-x) & \mbox{sigmoid} \label{equ_}\\
\dot{x}  =& 1/\tau             & \mbox{constant velocity (mimics the passage of time)} \label{equ_}\\
\f}

\image html sigmoid-svg.png "Exponential (blue) and sigmoid (purple) dynamical systems."
\image latex sigmoid-svg.pdf "Exponential (blue) and sigmoid (purple) dynamical systems." height=4cm

</em>

\section dyn_sys_second_order_systems Second-Order Systems 

The \b order of a dynamical system is the order of the highest derivative in the differential equation. For instance, \f$\dot{x} = -\alpha x\f$ is of order 1, because the derivative with the highest order (\f$\dot{x}\f$) has order 1. Such a system is known as a first-order system. All systems considered so far have been first-order systems, because the derivative with the highest order, i.e. \f$ \dot{x} \f$, has always been of order 1. 

\subsection dyn_sys_spring_damper Spring-Damper Systems 

An example of a second order system (which also has terms \f$ \ddot{x} \f$) is a spring-damper system (see http://en.wikipedia.org/wiki/Damped_spring-mass_system), where \f$k\f$ is the spring constant, \f$c\f$ is the damping coefficient, and \f$m\f$ is the mass:

\f{eqnarray*}{
m\ddot{x}=& -kx -c\dot{x}      & \mbox{spring-damper  (2nd order system)} \label{equ_}\\
\ddot{x}=& (-kx -c\dot{x})/m   &
\f}

\subsection dyn_sys_critical_damping Critical Damping 

A spring-damper system is called critically damped when it converges to the attractor as quickly as possible without overshooting, as the red plot in http://en.wikipedia.org/wiki/File:Damping_1.svg. This happens when \f$c = 2\sqrt{mk}\f$.


\subsection dyn_sys_rewrite_second_first Rewriting one 2nd Order Systems as two 1st Order Systems

For implementation purposes, it is more convenient to work only with 1st order systems. Fortunately, we can expand the state \f$ x \f$ into two components \f$ x = [y~z]^T\f$ with \f$ z = \dot{y}\f$, and rewrite the differential equation as follows:

\f$
\left[ \begin{array}{l} \dot{y} \\ \dot{z} \end{array} \right] = \left[ \begin{array}{l} z \\ (-ky -cz)/m \end{array} \right]
\f$

With this rewrite, the left term contains only first order derivatives, and the right term does not contain any derivatives. This is thus a first order system. Integrating such an expanded system is done just as one would integrate a dynamical system with a multi-dimensional state:

\subsection Implementation
<em>
The constructor DynamicalSystem::DynamicalSystem immediately converts second order systems into first order systems with an expanded state.

The function DynamicalSystem::dim() returns the size of the entire state vector \f$ x = [y~z]\f$, the function DynamicalSystem::dim_orig() return the size of only the \f$ y \f$ component. The attractor and initial state must always have the size returned by DynamicalSystem::dim_orig().
</em>

*/


