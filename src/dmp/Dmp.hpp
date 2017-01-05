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

#include "dynamicalsystems/DynamicalSystem.hpp"
#include "functionapproximators/Parameterizable.hpp"

#include "dmpbbo_io/EigenBoostSerialization.hpp"
#include <boost/serialization/assume_abstract.hpp>

#include <boost/random.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>

#include <set>

namespace DmpBbo {
  
// forward declaration
class FunctionApproximator;
class SpringDamperSystem;
class Trajectory;

/** \defgroup Dmps Dynamic Movement Primitives
 */

/** 
 * \brief Implementation of Dynamical Movement Primitives.
 * \ingroup Dmps
 */
class Dmp : public DynamicalSystem, public Parameterizable
{
public:
  
  /** Different types of DMPs that can be initialized. */
  enum DmpType { IJSPEERT_2002_MOVEMENT, KULVICIUS_2012_JOINING, COUNTDOWN_2013  };

  /** Different ways to scale the forcing term. */
  enum ForcingTermScaling { NO_SCALING, G_MINUS_Y0_SCALING, AMPLITUDE_SCALING };
  
  /**
   *  Initialization constructor.
   *  \param tau             Time constant
   *  \param y_init          Initial state
   *  \param y_attr          Attractor state
   *  \param alpha_spring_damper \f$\alpha\f$ in the spring-damper system of the dmp
   *  \param goal_system     Dynamical system to compute delayed goal
   *  \param phase_system    Dynamical system to compute the phase
   *  \param gating_system   Dynamical system to compute the gating term
   *  \param function_approximators Function approximators for the forcing term
   */
   Dmp(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr,
     std::vector<FunctionApproximator*> function_approximators,
     double alpha_spring_damper, DynamicalSystem* goal_system,
     DynamicalSystem* phase_system, DynamicalSystem* gating_system, 
     ForcingTermScaling scaling=NO_SCALING);
  
  /**
   *  Initialization constructor for Dmps of known dimensionality, but with unknown initial and
   *  attractor states.
   *  \param n_dims_dmp      Dimensionality of the DMP
   *  \param alpha_spring_damper \f$\alpha\f$ in the spring-damper system of the dmp
   *  \param goal_system     Dynamical system to compute delayed goal
   *  \param phase_system    Dynamical system to compute the phase
   *  \param gating_system   Dynamical system to compute the gating term
   *  \param function_approximators Function approximators for the forcing term
   */
   Dmp(int n_dims_dmp, std::vector<FunctionApproximator*> function_approximators, 
     double alpha_spring_damper, DynamicalSystem* goal_system,
     DynamicalSystem* phase_system, DynamicalSystem* gating_system,
     ForcingTermScaling scaling=NO_SCALING);
    
  /**
   *  Constructor that initializes the DMP with default dynamical systems.
   *  \param tau       Time constant
   *  \param y_init    Initial state
   *  \param y_attr    Attractor state
   *  \param function_approximators Function approximators for the forcing term
   *  \param dmp_type  The type of DMP, see Dmp::DmpType    
   */
  Dmp(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr, 
    std::vector<FunctionApproximator*> function_approximators, 
    DmpType dmp_type=KULVICIUS_2012_JOINING,   
    ForcingTermScaling scaling=NO_SCALING);

  
  /**
   *  Initialization constructor for Dmps of known dimensionality, but with unknown initial and
   *  attractor states. Initializes the DMP with default dynamical systems.
   *  \param n_dims_dmp      Dimensionality of the DMP
   *  \param function_approximators Function approximators for the forcing term
   *  \param dmp_type  The type of DMP, see Dmp::DmpType    
   */
  Dmp(int n_dims_dmp, std::vector<FunctionApproximator*> function_approximators,
    DmpType dmp_type=KULVICIUS_2012_JOINING, ForcingTermScaling scaling=NO_SCALING);      
   
  /**
   *  Initialization constructor for Dmps without a forcing term.
   *  \param tau             Time constant
   *  \param y_init          Initial state
   *  \param y_attr          Attractor state
   *  \param alpha_spring_damper \f$\alpha\f$ in the spring-damper system of the dmp
   *  \param goal_system     Dynamical system to compute delayed goal
   */
  Dmp(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr, double alpha_spring_damper, DynamicalSystem* goal_system);
  
  /** Destructor. */
  ~Dmp(void);
  
  /** Return a deep copy of this object 
   * \return A deep copy of this object
   */
  Dmp* clone(void) const;

  
  virtual void integrateStart(Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> xd) const;
  
  void differentialEquation(const Eigen::VectorXd& x, Eigen::Ref<Eigen::VectorXd> xd) const;
  
  /**
   * Return analytical solution of the system at certain times (and return forcing terms)
   *
   * \param[in]  ts  A vector of times for which to compute the analytical solutions
   * \param[out] xs  Sequence of state vectors. T x D or D x T matrix, where T is the number of times (the length of 'ts'), and D the size of the state (i.e. dim())
   * \param[out] xds Sequence of state vectors (rates of change). T x D or D x T matrix, where T is the number of times (the length of 'ts'), and D the size of the state (i.e. dim())
   * \param[out] forcing_terms The forcing terms for each dimension, for debugging purposes only.
   * \param[out] fa_output The output of the function approximators, for debugging purposes only.
   *
   * \remarks The output xs and xds will be of size D x T \em only if the matrix x you pass as an argument of size D x T. In all other cases (i.e. including passing an empty matrix) the size of x will be T x D. This feature has been added so that you may pass matrices of either size. 
   */
  void analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs, Eigen::MatrixXd& xds, Eigen::MatrixXd& forcing_terms, Eigen::MatrixXd& fa_output) const;
  
  /**
   * Return analytical solution of the system at certain times (and return forcing terms)
   *
   * \param[in]  ts  A vector of times for which to compute the analytical solutions
   * \param[out] xs  Sequence of state vectors. T x D or D x T matrix, where T is the number of times (the length of 'ts'), and D the size of the state (i.e. dim())
   * \param[out] xds Sequence of state vectors (rates of change). T x D or D x T matrix, where T is the number of times (the length of 'ts'), and D the size of the state (i.e. dim())
   * \param[out] forcing_terms The forcing terms for each dimension, for debugging purposes only.
   *
   * \remarks The output xs and xds will be of size D x T \em only if the matrix x you pass as an argument of size D x T. In all other cases (i.e. including passing an empty matrix) the size of x will be T x D. This feature has been added so that you may pass matrices of either size. 
   */
  inline void analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs, Eigen::MatrixXd& xds, Eigen::MatrixXd& forcing_terms) const
  {
    Eigen::MatrixXd fa_output;
    analyticalSolution(ts, xs, xds, forcing_terms, fa_output);
  }

  inline void analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs, Eigen::MatrixXd& xds) const
  {
    Eigen::MatrixXd forcing_terms, fa_output;
    analyticalSolution(ts, xs, xds, forcing_terms, fa_output);
  }

  /**
   * Return analytical solution of the system at certain times
   *
   * \param[in]  ts  A vector of times for which to compute the analytical solutions
   * \param[out] trajectory The computed states as a trajectory.
   */
  void analyticalSolution(const Eigen::VectorXd& ts, Trajectory& trajectory) const
  {
    Eigen::MatrixXd xs,  xds;
    analyticalSolution(ts, xs, xds);
    statesAsTrajectory(ts, xs, xds, trajectory);
  }

  /**
   * Return analytical solution of the system at certain times
   *
   * \param[in]  ts  A vector of times for which to compute the analytical solutions
   * \param[out] trajectory The computed states as a trajectory.
   * \param[out] forcing_terms The forcing terms
   */
  inline void analyticalSolution(const Eigen::VectorXd& ts, Trajectory& trajectory, Eigen::MatrixXd& forcing_terms) const
  {
    Eigen::MatrixXd xs,  xds;
    analyticalSolution(ts, xs, xds, forcing_terms);
    statesAsTrajectory(ts, xs, xds, trajectory);
  }

  
  
  
  /** Get the output of a DMP dynamical system as a trajectory.
   *  As a dynamical system, the state vector of a DMP contains the output of the goal, spring, 
   *  phase and gating system. What we are most interested in is the output of the spring system.
   *  This function extracts that information, and also computes the accelerations of the spring
   *  system, which are only stored implicitely in xd_in because second order systems are converted
   *  to first order systems with expanded state.
   *
   * \param[in] x_in  State vector over time (size n_time_steps X dim())
   * \param[in] xd_in State vector over time (rates of change)
   * \param[out] y_out  State vector over time (size n_time_steps X dim_orig())
   * \param[out] yd_out  State vector over time (rates of change)
   * \param[out] ydd_out  State vector over time (rates of change of rates of change)
   *  
   */
  virtual void statesAsTrajectory(const Eigen::MatrixXd& x_in, const Eigen::MatrixXd& xd_in, Eigen::MatrixXd& y_out, Eigen::MatrixXd& yd_out, Eigen::MatrixXd& ydd_out) const;
  
  /** Get the output of a DMP dynamical system as a trajectory.
   *  As a dynamical system, the state vector of a DMP contains the output of the goal, spring, 
   *  phase and gating system. What we are most interested in is the output of the spring system.
   *  This function extracts that information, and also computes the accelerations of the spring
   *  system, which are only stored implicitely in xd_in because second order systems are converted
   *  to first order systems with expanded state.
   *
   * \param[in] ts    A vector of times 
   * \param[in] x_in  State vector over time
   * \param[in] xd_in State vector over time (rates of change)
   * \param[out] trajectory Trajectory representation of the DMP state vector output.
   *  
   */
  virtual void statesAsTrajectory(const Eigen::VectorXd& ts, const Eigen::MatrixXd& x_in, const Eigen::MatrixXd& xd_in, Trajectory& trajectory) const;
  
  /**
   * Train a DMP with a trajectory.
   * \param[in] trajectory The trajectory with which to train the DMP.
   */
  void train(const Trajectory& trajectory);
      
  /**
   * Train a DMP with a trajectory, and write results to file
   * \param[in] trajectory The trajectory with which to train the DMP.
   * \param[in] save_directory The directory to which to save the results.
   * \param[in] overwrite Overwrite existing files in the directory above (default: false)
   */
  void train(const Trajectory& trajectory, std::string save_directory, bool overwrite=false);

  /**
   * Accessor function for the time constant.
   * \param[in] tau Time constant
   * We need to override DynamicalSystem::set_tau, because the DMP must also change the time
   * constant of all of its subsystems.
   */
  virtual void set_tau(double tau);

  /** Accessor function for the initial state of the system.
   *  \param[in] y_init Initial state of the system.
   * We need to override DynamicalSystem::set_initial_state, because the DMP must also change 
   * the initial state  of the goal system as well.
   */
  virtual void set_initial_state(const Eigen::VectorXd& y_init);
  
  /** Accessor function for the attractor state of the system. 
   *  \param[in] y_attr Attractor state of the system.
   */
  virtual void set_attractor_state(const Eigen::VectorXd& y_attr);
  
  
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

	std::string toString(void) const;
    
  //virtual bool isTrained(void) const;
  
  void getSelectableParameters(std::set<std::string>& selectable_values_labels) const;
  void setSelectedParameters(const std::set<std::string>& selected_values_labels);

  int getParameterVectorAllSize(void) const;
  void getParameterVectorAll(Eigen::VectorXd& values) const;
  void setParameterVectorAll(const Eigen::VectorXd& values);
  void getParameterVectorMask(const std::set<std::string> selected_values_labels, Eigen::VectorXi& selected_mask) const;

  /** Given a trajectory, compute the inputs and targets for the function approximators.
   * For a standard Dmp (such as the one in this class) the inputs will be the phase over time, and
   * the targets will be the forcing term (with the gating function factored out).
   * \param[in] trajectory Trajectory, e.g. a demonstration.
   * \param[out] fa_inputs_phase The inputs for the function approximators (phase signal)
   * \param[out] fa_targets The targets for the function approximators (forcing term)
   */
  void computeFunctionApproximatorInputsAndTargets(const Trajectory& trajectory, Eigen::VectorXd& fa_inputs_phase, Eigen::MatrixXd& fa_targets) const;
  
  /** Compute the outputs of the function approximators.
   * \param[in] phase_state The phase states for which the outputs are computed.
   * \param[out] fa_output The outputs of the function approximators.
   */
  virtual void computeFunctionApproximatorOutput(const Eigen::MatrixXd& phase_state, Eigen::MatrixXd& fa_output) const;
  
  /** Add a perturbation to the forcing term when computing the analytical solution.
   * This is only relevant for off-line experiments, i.e. not on a robot, for testing how
   * the system responds to perturbations. Does not affect the output of Dmp::differentialEquation(), only of Dmp::analyticalSolution().
   * \param[in] perturbation_standard_deviation Standard deviation of the normal distribution from which perturbations will be sampled.
   * 
   */
  void set_perturbation_analytical_solution(double perturbation_standard_deviation)
  {
    if (perturbation_standard_deviation>0.0)
    {
      boost::normal_distribution<> normal(0, perturbation_standard_deviation);
      analytical_solution_perturber_ = new boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >(rng, normal);
    }
    else
    {
      analytical_solution_perturber_ = NULL;
    }
  }
  
protected:

  /** Get a pointer to the function approximator for a certain dimension.
   * \param[in] i_dim Dimension for which to get the function approximator
   * \return Pointer to the function approximator.
   */
  inline FunctionApproximator* function_approximator(int i_dim) const
  {
    assert(i_dim<(int)function_approximators_.size());
    return function_approximators_[i_dim];
  }
   
  
private:
  /** @name Linear closed loop controller
   *  @{
   */ 
  /** Delayed goal system. Also see \ref sec_delayed_goal */
  DynamicalSystem* goal_system_;   
  /** Spring-damper system. Also see \ref page_dmp */
  SpringDamperSystem* spring_system_;
  /** @} */ // end of group_linear
  
  /** @name Non-linear open loop controller
   *  @{
   */ 
  /** System that determined the phase of the movement. */
  DynamicalSystem* phase_system_;
  /** System to gate the output of the function approximators. Starts at 1 and converges to 0. */
  DynamicalSystem* gating_system_;
  
  /** The function approximators, one for each dimension, in the forcing term. */
  std::vector<FunctionApproximator*> function_approximators_;
  
  /** How is the forcing term scaled? */
  ForcingTermScaling forcing_term_scaling_;
  
  /** Ranges of the trajectory (per dimension) for (optional) scaling of forcing term.  */
  Eigen::VectorXd trajectory_amplitudes_;

  /** @} */ // end of group_nonlinear
  
  /**
   *  Helper function for constructor.
   *  \param spring_system   Spring-damper system                 cf. Dmp::spring_system_
   *  \param goal_system     System to compute delayed goal,      cf. Dmp::damping_coefficient_
   *  \param phase_system    System to compute the phase,         cf. Dmp::phase_system_
   *  \param gating_system   System to compute the gating term,   cf. Dmp::gating_system_
   *  \param function_approximators Function approximators for the forcing term, cf. Dmp::function_approximators_
   */
  void initSubSystems(double alpha_spring_system, DynamicalSystem* goal_system,
    DynamicalSystem* phase_system, DynamicalSystem* gating_system);
  
  void initSubSystems(DmpType dmp_type);
  
  void initFunctionApproximators(std::vector<FunctionApproximator*> function_approximators);
  
  /** Boost's random number generator. Shared by all object instances. */
  static boost::mt19937 rng;
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > *analytical_solution_perturber_;
  
protected:
   Dmp(void) {};

private:
  /** Give boost serialization access to private members. */  
  friend class boost::serialization::access;
  
  /** Serialize class data members to boost archive. 
   * \param[in] ar Boost archive
   * \param[in] version Version of the class
   * See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/tutorial.html#simplecase
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version);

};

}

#include <boost/serialization/export.hpp>

/** Don't add version information to archives. */
BOOST_SERIALIZATION_ASSUME_ABSTRACT(DmpBbo::Dmp);
 
/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::Dmp,boost::serialization::object_serializable);

#endif // _DMP_H_


/** \page page_dmp Dynamical Movement Primitives Module

\section sec_dmp_introduction Introduction

The core idea behind dynamical movement primitives (DMPs) is to represent movement primitives as a combination of dynamical systems (please read \ref page_dyn_sys, if you haven't already done so). The state variables of the main dynamical system \f$ [\mathbf{y \dot{y} \ddot{y}} ]\f$ then represent trajectories for controlling, for instance, the 7 joints of a robot arm, or its 3D end-effector position. The attractor state is the end-point or \em goal of the movement.

The key advantage of DMPs is that they inherit the nice properties from linear dynamical systems (guaranteed convergence towards the attractor, robustness to perturbations, independence of time, etc) whilst allowing arbitrary (smooth) motions to be represented by adding a non-linear forcing term. This forcing term is often learned from demonstration, and subsequently improved through reinforcement learning.

DMPs were introduced in \cite ijspeert02movement, but in this section we follow largely the notation and description in \cite ijspeert13dynamical, but at a slower pace.

\em Historical \em remark. Recently, the term "dynamicAL movement primitives" is preferred over "dynamic movement primitives". The newer term makes the relation to dynamicAL systems more clear, and avoids confusion about whether the output of "dynamical movement primitives" is in kinematic or dynamic space (it is usually in kinematic space).

\em Remark. This documentation and code focusses only on discrete movement primitives. For rythmic movement primitives, we refer to \cite ijspeert13dynamical.


\section sec_core Basic Point-to-Point Movements: A Critically Damped Spring-Damper System

At the heart of the DMP lies a spring-damper system, as described in \ref dyn_sys_spring_damper. In DMP papers, the notation of the spring-damper system is usually a bit different:
\f{eqnarray*}{
m\ddot{y}  =& -ky -c\dot{y} & \mbox{spring-damper system, ``traditional notation''} \\
m\ddot{y}  =& c(-\frac{k}{c}y - \dot{y})\\
\tau\ddot{y}  =& \alpha(-\beta y - \dot{y})      & \mbox{with } \alpha=c,~~\beta = \frac{k}{c},~~m=\tau\\
\tau\ddot{y}  =& \alpha(-\beta (y-y^g) - \dot{y})& \mbox{with attractor } y^g\\
\tau\ddot{y}  =& \alpha(\beta (y^g-y) - \dot{y})& \mbox{typical DMP notation for spring-damper system}\\
\f}

In the last two steps, we change the attractor state from 0 to \f$y^g\f$, where \f$y^g\f$ is the goal of the movement.

To avoid overshooting or slow convergence towards \f$y^g\f$, we prefer to have a \em critically \em damped spring-damper system for the DMP. For such systems \f$c = 2\sqrt{mk}\f$ must hold, see \ref dyn_sys_critical_damping. In our notation this becomes \f$\alpha = 2\sqrt{\alpha\beta}\f$, which leads to \f$\beta = \alpha/4\f$. This determines the value of \f$\beta\f$ for a given value of \f$\alpha\f$ in DMPs. The influence of \f$\alpha\f$ is illustrated in the first figure in \ref sec_dyn_sys_intro.

Rewriting the second order dynamical system as a first order system (see \ref dyn_sys_rewrite_second_first) with expanded state \f$ [z~y]\f$ yields:

\f{eqnarray*}{
\left[ \begin{array}{l} {\dot{z}} \\ {\dot{y}} \end{array} \right] = \left[ \begin{array}{l} (\alpha (\beta({y}^{g}-{y})-{z}))/\tau \\ {z}/\tau  \end{array} \right]  \mbox{~~~~with init. state~} \left[ \begin{array}{l} 0 \\ y_0  \end{array} \right] \mbox{~and attr. state~} \left[ \begin{array}{l} {0} \\ {y}^g \end{array} \right]
\f}

Please note that in the implementation, the state is implemented as  \f$ [y~z]\f$. The order is inconsequential, but we use the notation above (\f$[z~y]\f$) throughout the rest of this tutorial section, for consistency with the DMP literature.


\section sec_forcing Arbitrary Smooth Movements: the Forcing Term

The representation described in the previous section has some nice properties in terms of 
\ref sec_dyn_sys_convergence
, \ref sec_dyn_sys_perturbations
, and \ref sec_dyn_sys_autonomy, but it can only represent very simple movements. To achieve more complex movements, we add a time-dependent forcing term to the spring-damper system. The spring-damper systems and forcing term are together known as a \em transformation \em system.

\f{eqnarray*}{
\left[ \begin{array}{l} {\dot{z}} \\ {\dot{y}} \end{array} \right] = \left[ \begin{array}{l} (\alpha (\beta({y}^{g}-{y})-{z}) + f(t))/\tau \\ {z}/\tau  \end{array} \right]   \mbox{~~~~with init. state~} \left[ \begin{array}{l} 0 \\ y_0  \end{array} \right] \mbox{~and attr. state~} \left[ \begin{array}{l} {?} \\ {y}^g \end{array} \right]
\f}

The forcing term is an open loop controller, i.e. it depends only on time. By modifying the acceleration profile of the movement with a forcing term, arbitrary smooth movements can be achieved. The function \f$ f(t)\f$ is usually a function approximator, such as locally weighted regression (LWR) or locally weighted projection regression (LWPR), see \ref page_func_approx. The graph below shows an example of a forcing term implemented with LWR with random weights for the basis functions.

\image html dmp_forcing_terms-svg.png "A non-linear forcing term enable more complex trajectories to be generated (these DMPs use a goal system and an exponential gating term)."
\image latex dmp_forcing_terms-svg.pdf "A non-linear forcing term enable more complex trajectories to be generated (these DMPs use a goal system and an exponential gating term)." height=4cm

\subsection sec_forcing_convergence Ensuring Convergence to 0 of the Forcing Term: the Gating System

Since we add a forcing term to the dynamical system, we can no longer guarantee that the system will converge towards \f$ x^g \f$; perhaps the forcing term continually pushes it away \f$ x^g \f$ (perhaps it doesn't, but the point is that we cannot \em guarantee that it \em always doesn't). That is why there is a question mark in the attractor state in the equation above.
To guarantee that the movement will always converge towards the attractor \f$ x^g \f$, we need to ensure that the forcing term decreases to 0 towards the end of the movement. To do so, a gating term is added, which is 1 at the beginning of the movement, and 0 at the end. This gating term itself is determined by, of course, a dynamical system. In \cite ijspeert02movement, it was suggested to use an exponential system. We add this extra system to our dynamical system by expanding the state as follows:

\f{eqnarray*}{
\dot{x} = \left[ \begin{array}{l} {\dot{z}} \\ {\dot{y}} \\ {\dot{x}} \end{array} \right] = \left[ \begin{array}{l} (\alpha_y (\beta_y({y}^{g}-{y})-{z}) + x\cdot f(t))/\tau \\ {z}/\tau \\ -\alpha_x x/\tau  \end{array} \right] \mbox{~~~~with init. state~} \left[ \begin{array}{l} 0 \\ y_0 \\ 1 \end{array} \right] \mbox{~and attr. state~} \left[ \begin{array}{l} {0} \\ {y}^g \\ 0 \end{array} \right]
\f}

\subsection sec_forcing_autonomy Ensuring Autonomy of the Forcing Term: the Phase System

By introducing the dependence of the forcing term \f$ f(t)\f$ on time \f$ t \f$ the overall system is no longer autonomous. To achieve independence of time, we therefore let \f$ f \f$ be a function of the state of an (autonomous) dynamical system rather than of \f$ t \f$. This system represents the \em phase of the movement. \cite ijspeert02movement suggested to use the same dynamical system for the gating and phase, and use the term \em canonical \em system to refer this joint gating/phase system. Thus the phase of the movement starts at 1, and converges to 0 towards the end of the movement, just like the gating system. The new formulation now is (the only difference is \f$ f(x)\f$ instead of \f$ f(t)\f$):

\f{eqnarray*}{
\left[ \begin{array}{l} {\dot{z}} \\ {\dot{y}} \\ {\dot{x}} \end{array} \right] = \left[ \begin{array}{l} (\alpha_y (\beta_y({y}^{g}-{y})-{z}) + x\cdot f(x))/\tau \\ {z}/\tau \\ -\alpha_x x/\tau  \end{array} \right] \mbox{~~~~with init. state~} \left[ \begin{array}{l} 0 \\ y_0 \\ 1 \end{array} \right] \mbox{~and attr. state~} \left[ \begin{array}{l} {0} \\ {y}^g \\ 0 \end{array} \right]
\f}

\todo Discuss goal-dependent scaling, i.e. \f$ f(t)s(x^g-x_0) \f$?


\subsection sec_multidim_dmp Multi-dimensional Dynamic Movement Primitives

Since DMPs usually have multi-dimensional states (e.g. one output \f$ {\mathbf{y}}_{d=1\dots D}\f$ for each of the \f$ D \f$ joints), it is more accurate to use bold fonts for the state variables (except the gating/phase system, because it is always 1D) so that they represent vectors:

\f{eqnarray*}{
\left[ \begin{array}{l} {\dot{\mathbf{z}}} \\ {\dot{\mathbf{y}}} \\ {\dot{x}} \end{array} \right] = \left[ \begin{array}{l} (\alpha_y (\beta_y({\mathbf{y}}^{g}-\mathbf{y})-\mathbf{z}) + x\cdot f(x))/\tau \\ \mathbf{z}/\tau \\ -\alpha_x x/\tau  \end{array} \right] \mbox{~~~~with init. state~} \left[ \begin{array}{l} \mathbf{0} \\ \mathbf{z}_0 \\ 1 \end{array} \right] \mbox{~and attr. state~} \left[ \begin{array}{l} \mathbf{0} \\ \mathbf{y}^g \\ 0 \end{array} \right]
\f}

So far, the graphs have shown 1-dimensional systems. To generate D-dimensional trajectories for, for instance, the 7 joints of an arm or the 3D position of its end-effector, we simply use D transformation systems. A key principle in DMPs is to use one and the same phase system for all of the transformation systems, to ensure that the output of the transformation systems are synchronized in time. The image below show the evolution of all the dynamical systems involved in integrating a multi-dimensional DMP.

\image html dmpplot_ijspeert2002movement-svg.png "The various dynamical systems and forcing terms in multi-dimensional DMPs."
\image latex dmpplot_ijspeert2002movement-svg.pdf "The various dynamical systems and forcing terms in multi-dimensional DMPs." height=8cm

<em>

\subsection Implementation

Since a Dynamical Movement Primitive is a dynamical system, the Dmp class derives from the DynamicalSystem class. It overrides the virtual function DynamicalSystem::integrateStart(). Integrating the DMP numerically (Euler or 4th order Runge-Kutta) is done with the generic DynamicalSystem::integrateStep() function. It also implements the pure virtual function DynamicalSystem::analyticalSolution(). Because a DMP cannot be solved analytically (we cannot write it in closed form due to the arbitrary forcing term), calling Dmp::analyticalSolution() in fact performs a numerical Euler integration (although the linear subsystems (phase, gating, etc.) are analytically solved because this is faster computationally).

Please note that in this tutorial we have used the notation \f$[z~y]\f$ for consistency with the DMP literature. In the C++ implementation, the order is rather \f$[y~z]\f$.

\em Remark. Dmp inherits the function DynamicalSystem::integrateStep() from the DynamicalSystem class. DynamicalSystem::integrateStep() uses either Euler integration, or 4-th order Runge-Kutta.  The latter is more accurate, but requires 4 calls of DynamicalSystem::differentialEquation() instead of 1). Which one is used can be set with DynamicalSystem::set_integration_method(). To numerically integrate a dynamical system, one must carefully choose the integration time dt. Choosing it too low leads to inaccurate integration, and the numerical integration will diverge from the 'true' solution acquired through analytical solution. See http://en.wikipedia.org/wiki/Euler%27s_method for examples. Choosing dt depends entirely on the time-scale (seconds vs. years) and parameters of the dynamical system (time constant, decay parameters). For DMPs, which are expected to take between 0.5-10 seconds, dt is usually chosen to be in the range 0.01-0.001.
</em>


\section sec_dmp_alternative Alternative Systems for Gating, Phase and Goals 

\subsection sec_dmp_sigmoid_gating Gating: Sigmoid System

A disadvantage of using an exponential system as a gating term is that the gating decreases very quickly in the beginning. Thus, the output of the function approximator \f$ f(x) \f$ needs to be very high towards the end of the movement if it is to have any effect at all. This leads to scaling issues when training the function approximator.

Therefore, sigmoid systems have more recently been proposed \cite kulvicius12joining as a gating system. This leads to the following DMP formulation (since the gating and phase system are no longer shared, we introduce a new state variable \f$ v \f$ for the gating term:

\f{eqnarray*}{
\left[ \begin{array}{l} {\dot{\mathbf{z}}} \\ {\dot{\mathbf{y}}} \\ {\dot{x}}  \\ {\dot{v}} \end{array} \right] = \left[ \begin{array}{l} (\alpha_y (\beta_y({\mathbf{y}}^{g}-\mathbf{y})-\mathbf{z}) + v\cdot f(x))/\tau \\ \mathbf{z}/\tau \\ -\alpha_x x/\tau \\ -\alpha_v v (1-v/v_{\mbox{\scriptsize max}}) \end{array} \right] \mbox{~~~~with init. state~} \left[ \begin{array}{l} \mathbf{0} \\ \mathbf{y}_0 \\ 1 \\ 1 \end{array} \right]
\mbox{~and attr. state~} \left[ \begin{array}{l} \mathbf{0} \\ \mathbf{y}^g \\ 0 \\ 0 \end{array} \right]
\f}

where the term \f$ v_{\mbox{\scriptsize max}}\f$ is determined by \f$\tau \f$

\subsection sec_dmp_phase Phase: Constant Velocity System

In practice, using an exponential phase system may complicate imitation learning of the function approximator \f$ f \f$, because samples are not equidistantly spaced in time. Therefore, we introduce a dynamical system that mimics the properties of the phase system described in \cite kulvicius12joining, whilst allowing for a more natural integration in the DMP formulation, and thus our code base. This system starts at 0, and has a constant velocity of \f$1/\tau\f$, which means the system reaches 1 when \f$t=\tau\f$. When this point is reached, the velocity is set to 0. 

\f{eqnarray*}{
\dot{x} =& 1/\tau \mbox{~if~} x < 1   & \\
         & 0 \mbox{~if~} x>1 \\
\f}

This, in all honesty, is a bit of a hack, because it leads to a non-smooth acceleration profile. However, its properties as an input to the function approximator are so advantageous that we have designed it in this way (the implementation of this system is in the TimeSystem class).


\image html phase_systems-svg.png "Exponential and constant velocity dynamical systems as the 1D phase for a dynamical movement primitive."
\image latex phase_systems-svg.pdf "Exponential and constant velocity dynamical systems as the 1D phase for a dynamical movement primitive." height=4cm

With the constant velocity dynamical system the DMP formulation becomes:

\f{eqnarray*}{
\left[ \begin{array}{l} {\dot{\mathbf{z}}} \\ {\dot{\mathbf{y}}} \\ {\dot{x}}  \\ {\dot{v}} \end{array} \right] = \left[ \begin{array}{l} (\alpha_y (\beta_y({\mathbf{y}}^{g}-\mathbf{y})-\mathbf{z}) + v\cdot f(x))/\tau \\ \mathbf{z}/\tau \\ 1/\tau \\ -\alpha_v v (1-v/v_{\mbox{\scriptsize max}}) \end{array} \right] \mbox{~~~~with init. state~} \left[ \begin{array}{l} \mathbf{0} \\ \mathbf{y}_0 \\ 0 \\ 1 \end{array} \right]
\mbox{~and attr. state~} \left[ \begin{array}{l} \mathbf{0} \\ \mathbf{y}^g \\ 1 \\ 0 \end{array} \right]
\f}

\subsection sec_delayed_goal Zero Initial Accelerations: the Delayed Goal System

Since the spring-damper system leads to high initial accelerations (see the graph to the right below), which is usually not desirable for robots, it was suggested to move the attractor of the system from the initial state \f$ y_0 \f$ to the goal state \f$ y^g \f$  \em during the movement \cite kulvicius12joining. This delayed goal attractor \f$ y^{g_d} \f$ itself is represented as an exponential dynamical system that starts at \f$ y_0 \f$, and converges to \f$ y^g \f$ (in early versions of DMPs, there was no delayed goal system, and \f$ y^{g_d} \f$ was simply equal to \f$ y^g \f$ throughout the movement). The combination of these two systems, listed below, leads to a movement that starts and ends with 0 velocities and accelerations, and approximately has a bell-shaped velocity profile. This representation is thus well suited to generating human-like point-to-point movements, which have similar properties.

\f{eqnarray*}{
\left[ \begin{array}{l} {\dot{\mathbf{z}}} \\ {\dot{\mathbf{y}}} \\ {\dot{\mathbf{y}}^{g_d}} \\ {\dot{x}}  \\ {\dot{v}} \end{array} \right] = \left[ \begin{array}{l} (\alpha_y (\beta_y({\mathbf{y}}^{g_d}-\mathbf{y})-\mathbf{z}) + v\cdot f(x))/\tau \\ \mathbf{z}/\tau \\ -\alpha_g({\mathbf{y}^g-\mathbf{y}^{g_d}}) \\ 1/\tau \\ -\alpha_v v (1-v/v_{\mbox{\scriptsize max}}) \end{array} \right] \mbox{~~~~with init. state~} \left[ \begin{array}{l} \mathbf{0} \\ \mathbf{y}_0 \\ \mathbf{y}_0 \\ 0 \\ 1 \end{array} \right]
\mbox{~and attr. state~} \left[ \begin{array}{l} \mathbf{0} \\ \mathbf{y}^g \\ \mathbf{y}^g \\ 1 \\ 0 \end{array} \right]
\f}


\image html dmp_and_goal_system-svg.png "A first dynamical movement primitive, with and without a delayed goal system (left: state variable, center: velocities, right: accelerations."
\image latex dmp_and_goal_system-svg.pdf "A first dynamical movement primitive, with and without a delayed goal system (left: state variable, center: velocities, right: accelerations." height=4cm


In my experience, this DMP formulation is the best for learning human-like point-to-point movements (bell-shaped velocity profile, approximately zero velocities and accelerations at beginning and start of the movement), and generates nice normalized data for the function approximator without scaling issues (an exact empirical evaluation is on the stack...). The image below shows the interactions between the spring-damper system, delayed goal system, phase system and gating system.


\image html dmpplot_kulvicius2012joining-svg.png "The various dynamical systems and forcing terms in multi-dimensional DMPs."
\image latex dmpplot_kulvicius2012joining-svg.pdf "The various dynamical systems and forcing terms in multi-dimensional DMPs." height=7cm


\section sec_dmp_issues Known Issues

\todo Known Issues

\li Scaling towards novel goals

\section sec_dmp_summary Summary

The core idea in dynamical movement primitives is to combine dynamical systems, which have nice properties in terms of convergence towards the goal, robustness to perturbations, and independence of time, with function approximators, which allow for the generation of arbitrary (smooth) trajectories. The key enabler to this approach is to gate the output of the function approximator with a gating system, which is 1 at the beginning of the movement, and 0 towards the end.

Further enhancements can be made by making the system autonomous (by using the output of a phase system rather than time as an input to the function approximator), or having initial velocities and accelerations of 0 (by using a delayed goal system).

Multi-dimensional DMPs are achieved by using multi-dimensional dynamical systems, and learning one function approximator for each dimension. Synchronization of the different dimensions is ensure by coupling them with only \em one phase system.

*/
