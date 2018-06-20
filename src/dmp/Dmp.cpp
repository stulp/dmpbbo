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

#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include "dmp/Dmp.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::Dmp);

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <eigen3/Eigen/Core>

#include "dmp/Trajectory.hpp"
#include "functionapproximators/FunctionApproximator.hpp"
#include "dynamicalsystems/SpringDamperSystem.hpp"
#include "dynamicalsystems/ExponentialSystem.hpp"
#include "dynamicalsystems/TimeSystem.hpp"
#include "dynamicalsystems/SigmoidSystem.hpp"

#include "dmpbbo_io/EigenBoostSerialization.hpp"
#include "dmpbbo_io/BoostSerializationToString.hpp"


using namespace std;
using namespace Eigen;

namespace DmpBbo {

/** Extracts all variables of the spring-damper system from a state vector, e.g. state.SPRING */ 
#define SPRING    segment(0*dim_orig()+0,2*dim_orig())
/** Extracts first order variables of the spring-damper system from a state vector, e.g. state.SPRINGM_Y */ 
#define SPRING_Y  segment(0*dim_orig()+0,dim_orig())
/** Extracts second order variables of the spring-damper system from a state vector, e.g. state.SPRING_Z */ 
#define SPRING_Z  segment(1*dim_orig()+0,dim_orig())
/** Extracts all variables of the goal from a state vector, e.g. state.GOAL */ 
#define GOAL      segment(2*dim_orig()+0,dim_orig())
/** Extracts the phase variable (1-D) from a state vector, e.g. state.PHASE */ 
#define PHASE     segment(3*dim_orig()+0,       1)
/** Extracts all variables of the gating system from a state vector, e.g. state.GATING */ 
#define GATING    segment(3*dim_orig()+1,       1)

/** Extracts first T (time steps) state vectors of the spring-damper system , e.g. states.SPRING(100) */ 
#define SPRINGM(T)    block(0,0*dim_orig()+0,T,2*dim_orig())
/** Extracts first T (time steps) state vectors of the spring-damper system , e.g. states.SPRINGM_Y(100) */ 
#define SPRINGM_Y(T)  block(0,0*dim_orig()+0,T,dim_orig())
/** Extracts first T (time steps) state vectors of the spring-damper system , e.g. states.SPRINGM_Z(100) */ 
#define SPRINGM_Z(T)  block(0,1*dim_orig()+0,T,dim_orig())
/** Extracts first T (time steps) state vectors of the goal system, e.g. states.GOALM(100) */ 
#define GOALM(T)      block(0,2*dim_orig()+0,T,dim_orig())
/** Extracts first T (time steps) states of the phase system, e.g. states.PHASEM(100) */ 
#define PHASEM(T)     block(0,3*dim_orig()+0,T,       1)
/** Extracts first T (time steps) state vectors of the gating system, e.g. states.GATINGM(100) */ 
#define GATINGM(T)    block(0,3*dim_orig()+1,T,       1)

boost::mt19937 Dmp::rng = boost::mt19937(getpid() + time(0));

Dmp::Dmp(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr, 
         std::vector<FunctionApproximator*> function_approximators,
         double alpha_spring_damper, 
         DynamicalSystem* goal_system,
         DynamicalSystem* phase_system, 
         DynamicalSystem* gating_system,     
         ForcingTermScaling scaling)
  : DynamicalSystem(1, tau, y_init, y_attr, "name"),
  goal_system_(goal_system),
  phase_system_(phase_system), gating_system_(gating_system), 
  forcing_term_scaling_(scaling)
{
  initSubSystems(alpha_spring_damper, goal_system, phase_system, gating_system);
  initFunctionApproximators(function_approximators);
}

  
Dmp::Dmp(int n_dims_dmp, std::vector<FunctionApproximator*> function_approximators, 
   double alpha_spring_damper, DynamicalSystem* goal_system,
   DynamicalSystem* phase_system, DynamicalSystem* gating_system,     
   ForcingTermScaling scaling)
  : DynamicalSystem(1, 1.0, VectorXd::Zero(n_dims_dmp), VectorXd::Ones(n_dims_dmp), "name"),
  goal_system_(goal_system),
  phase_system_(phase_system), gating_system_(gating_system), function_approximators_(function_approximators),
  forcing_term_scaling_(scaling)
{
  initSubSystems(alpha_spring_damper, goal_system, phase_system, gating_system);
  initFunctionApproximators(function_approximators);
}
    
Dmp::Dmp(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr, 
         std::vector<FunctionApproximator*> function_approximators, 
         DmpType dmp_type,     
         ForcingTermScaling scaling)
  : DynamicalSystem(1, tau, y_init, y_attr, "name"),
    forcing_term_scaling_(scaling)
{  
  initSubSystems(dmp_type);
  initFunctionApproximators(function_approximators);
}
  
Dmp::Dmp(int n_dims_dmp, 
         std::vector<FunctionApproximator*> function_approximators, 
         DmpType dmp_type, ForcingTermScaling scaling)
  : DynamicalSystem(1, 1.0, VectorXd::Zero(n_dims_dmp), VectorXd::Ones(n_dims_dmp), "name"),
    forcing_term_scaling_(scaling)
{
  initSubSystems(dmp_type);
  initFunctionApproximators(function_approximators);
}

Dmp::Dmp(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr, double alpha_spring_damper, DynamicalSystem* goal_system) 
  : DynamicalSystem(1, tau, y_init, y_attr, "name"), forcing_term_scaling_(NO_SCALING)
{
  
  VectorXd one_1 = VectorXd::Ones(1);
  VectorXd one_0 = VectorXd::Zero(1);
  DynamicalSystem* phase_system  = new ExponentialSystem(tau,one_1,one_0,4);
  DynamicalSystem* gating_system = new ExponentialSystem(tau,one_1,one_0,4); 
  initSubSystems(alpha_spring_damper, goal_system, phase_system, gating_system);

  vector<FunctionApproximator*> function_approximators(y_init.size());    
  for (int dd=0; dd<y_init.size(); dd++)
    function_approximators[dd] = NULL;
  initFunctionApproximators(function_approximators);  
}

void Dmp::initSubSystems(DmpType dmp_type)
{
  VectorXd one_1 = VectorXd::Ones(1);
  VectorXd one_0 = VectorXd::Zero(1);
  
  DynamicalSystem *goal_system=NULL;
  DynamicalSystem *phase_system=NULL;
  DynamicalSystem *gating_system=NULL;
  if (dmp_type==IJSPEERT_2002_MOVEMENT)
  {
    goal_system   = NULL;
    phase_system  = new ExponentialSystem(tau(),one_1,one_0,4);
    gating_system = new ExponentialSystem(tau(),one_1,one_0,4); 
  }                                                          
  else if (dmp_type==KULVICIUS_2012_JOINING || dmp_type==COUNTDOWN_2013)
  {
    goal_system   = new ExponentialSystem(tau(),initial_state(),attractor_state(),15);
    gating_system = new SigmoidSystem(tau(),one_1,-20,0.9*tau()); 
    bool count_down = (dmp_type==COUNTDOWN_2013);    
    phase_system  = new TimeSystem(tau(),count_down);
  }
  
  double alpha_spring_damper = 20;
  
  initSubSystems(alpha_spring_damper, goal_system, phase_system, gating_system);
}

void Dmp::initSubSystems(double alpha_spring_damper, DynamicalSystem* goal_system,
  DynamicalSystem* phase_system, DynamicalSystem* gating_system)
{

  // Make room for the subsystems
  set_dim(3*dim_orig()+2);
    
  spring_system_ = new SpringDamperSystem(tau(),initial_state(),attractor_state(),alpha_spring_damper);  
  spring_system_->set_name(name()+"_spring-damper");

  goal_system_ = goal_system;
  if (goal_system!=NULL)
  {
    assert(goal_system->dim()==dim_orig());
    // Initial state of the goal system is that same as that of the DMP
    goal_system_->set_initial_state(initial_state());
    goal_system_->set_name(name()+"_goal");
  }

  phase_system_ = phase_system;
  phase_system_->set_name(name()+"_phase");

  gating_system_ = gating_system;
  gating_system_->set_name(name()+"_gating");
  
  // Pre-allocate memory for real-time execution
  attractor_state_prealloc_ = VectorXd(dim_orig());
  initial_state_prealloc_ = VectorXd(dim_orig());
  fa_outputs_one_prealloc_ = MatrixXd(1,dim_orig());
  fa_outputs_prealloc_ = MatrixXd(1,dim_orig());
  fa_output_prealloc_ = MatrixXd(1,dim_orig()); 
  forcing_term_prealloc_ = VectorXd(dim_orig());
  g_minus_y0_prealloc_ = VectorXd(dim_orig());
  
}

void Dmp::set_damping_coefficient(double damping_coefficient)
{
  spring_system_->set_damping_coefficient(damping_coefficient); 
}
void Dmp::set_spring_constant(double spring_constant) {
  spring_system_->set_spring_constant(spring_constant); 
}
  

void Dmp::initFunctionApproximators(vector<FunctionApproximator*> function_approximators)
{
  if (function_approximators.empty())
    return;
  
  assert(dim_orig()==(int)function_approximators.size());
  
  function_approximators_ = vector<FunctionApproximator*>(function_approximators.size());
  for (unsigned int dd=0; dd<function_approximators.size(); dd++)
  {
    if (function_approximators[dd]==NULL)
      function_approximators_[dd] = NULL;
    else
      function_approximators_[dd] = function_approximators[dd]->clone();
  }

}

Dmp::~Dmp(void)
{
  delete goal_system_;   
  delete spring_system_;
  delete phase_system_;
  delete gating_system_;
  for (unsigned int ff=0; ff<function_approximators_.size(); ff++)
    delete (function_approximators_[ff]);
}

Dmp* Dmp::clone(void) const {
  vector<FunctionApproximator*> function_approximators;
  for (unsigned int ff=0; ff<function_approximators_.size(); ff++)
    function_approximators.push_back(function_approximators_[ff]->clone());
  
  return new Dmp(tau(), initial_state(), attractor_state(), function_approximators,
   spring_system_->damping_coefficient(), goal_system_->clone(),
   phase_system_->clone(), gating_system_->clone());
}


void Dmp::integrateStart(Ref<VectorXd> x, Ref<VectorXd> xd) const
{
  assert(x.size()==dim());
  assert(xd.size()==dim());
  
  x.fill(0);  
  xd.fill(0);  
  
  // Start integrating goal system if it exists
  if (goal_system_==NULL)
  {
    // No goal system, simply set goal state to attractor state
    x.GOAL = attractor_state();
    xd.GOAL.fill(0);
  }
  else
  {
    // Goal system exists. Start integrating it.
    goal_system_->integrateStart(x.GOAL,xd.GOAL);
  }
  
    
  // Set the attractor state of the spring system
  spring_system_->set_attractor_state(x.GOAL);
  
  // Start integrating all futher subsystems
  spring_system_->integrateStart(x.SPRING, xd.SPRING);
  phase_system_->integrateStart(  x.PHASE,  xd.PHASE);
  gating_system_->integrateStart(x.GATING, xd.GATING);

  // Add rates of change
  differentialEquation(x,xd);
  
}

/*
bool Dmp::isTrained(void) const
{
  for (int i_dim=0; i_dim<dim_orig(); i_dim++)
    if (function_approximators_[i_dim]!=NULL)
      return false;
    
  for (int i_dim=0; i_dim<dim_orig(); i_dim++)
    if (function_approximators_[i_dim]->isTrained()) 
      return false;
    
  return true;
}
*/

void Dmp::computeFunctionApproximatorOutput(const Ref<const MatrixXd>& phase_state, MatrixXd& fa_output) const
{
  int T = phase_state.rows();
  fa_output.resize(T,dim_orig());
  fa_output.fill(0.0);
  
  if (T>1) {
    fa_outputs_prealloc_.resize(T,dim_orig());
  }
  
  for (int i_dim=0; i_dim<dim_orig(); i_dim++)
  {
    if (function_approximators_[i_dim]!=NULL)
    {
      if (function_approximators_[i_dim]->isTrained()) 
      {
        if (T==1)
        {
          function_approximators_[i_dim]->predict(phase_state,fa_outputs_one_prealloc_);
          fa_output.col(i_dim) = fa_outputs_one_prealloc_;
        }
        else
        {
          function_approximators_[i_dim]->predict(phase_state,fa_outputs_prealloc_);
          fa_output.col(i_dim) = fa_outputs_prealloc_;
        }
      }
    }
  }
}

void Dmp::differentialEquation(
  const Eigen::Ref<const Eigen::VectorXd>& x, 
  Eigen::Ref<Eigen::VectorXd> xd) const
{
  
  ENTERING_REAL_TIME_CRITICAL_CODE
  
  attractor_state(attractor_state_prealloc_);  
  if (goal_system_==NULL)
  {
    // If there is no dynamical system for the delayed goal, the goal is
    // simply the attractor state
    spring_system_->set_attractor_state(attractor_state_prealloc_);
    // with zero change
    xd.GOAL.fill(0);
  } 
  else
  {
    // Integrate goal system and get current goal state
    goal_system_->set_attractor_state(attractor_state_prealloc_);
    goal_system_->differentialEquation(x.GOAL, xd.GOAL);
    // The goal state is the attractor state of the spring-damper system
    spring_system_->set_attractor_state(x.GOAL);
    
  }

  
  // Integrate spring damper system
  // Forcing term is added to spring_state later
  spring_system_->differentialEquation(x.SPRING, xd.SPRING);

  
  // Non-linear forcing term
  phase_system_->differentialEquation(x.PHASE, xd.PHASE);
  gating_system_->differentialEquation(x.GATING, xd.GATING);


  //MatrixXd phase_state(1,1);
  //phase_state = x.PHASE;
  computeFunctionApproximatorOutput(x.PHASE, fa_output_prealloc_); 

  // Gate the output of the function approximators
  int t0 = 0; 
  double gating = (x.GATING)[0];
  forcing_term_prealloc_ = gating*fa_output_prealloc_.row(t0);
  
  
  // Scale the forcing term, if necessary
  if (forcing_term_scaling_==G_MINUS_Y0_SCALING)
  {
    initial_state(initial_state_prealloc_);  
    g_minus_y0_prealloc_ = (attractor_state_prealloc_-initial_state_prealloc_).transpose();
    forcing_term_prealloc_ = forcing_term_prealloc_.array()*g_minus_y0_prealloc_.array();
  }
  else if (forcing_term_scaling_==AMPLITUDE_SCALING)
  {
    forcing_term_prealloc_ = forcing_term_prealloc_.array()*trajectory_amplitudes_.array();
  }

  // Add forcing term to the ZD component of the spring state
  xd.SPRING_Z = xd.SPRING_Z + forcing_term_prealloc_/tau();

  EXITING_REAL_TIME_CRITICAL_CODE

}

void Dmp::statesAsTrajectory(const Eigen::MatrixXd& x_in, const Eigen::MatrixXd& xd_in, Eigen::MatrixXd& y_out, Eigen::MatrixXd& yd_out, Eigen::MatrixXd& ydd_out) const
{
  int n_time_steps = x_in.rows(); 
  y_out  = x_in.SPRINGM_Y(n_time_steps);
  yd_out = xd_in.SPRINGM_Y(n_time_steps);
  ydd_out = xd_in.SPRINGM_Z(n_time_steps)/tau();
  // MatrixXd z_out, zd_out;
  // z_out  = x_in.SPRINGM_Z(n_time_steps);
  // zd_out = xd_in.SPRINGM_Z(n_time_steps);
  // Divide by tau to go from z to y space
  // yd = z_out/obj.tau;
  // ydd_out = zd_out/tau();
}


void Dmp::statesAsTrajectory(const Eigen::VectorXd& ts, const Eigen::MatrixXd& x_in, const Eigen::MatrixXd& xd_in, Trajectory& trajectory) const {
  int n_time_steps = ts.rows();
#ifndef NDEBUG // Variables below are only required for asserts; check for NDEBUG to avoid warnings.
  int n_dims       = x_in.cols();
#endif
  assert(n_time_steps==x_in.rows());
  assert(n_time_steps==xd_in.rows());
  assert(n_dims==xd_in.cols());

  // Left column is time
  Trajectory new_trajectory(
    ts,
    // y_out (see function above)
    x_in.SPRINGM_Y(n_time_steps),
    // yd_out (see function above)
    xd_in.SPRINGM_Y(n_time_steps),
    // ydd_out (see function above)
    xd_in.SPRINGM_Z(n_time_steps)/tau()
  );
  
  trajectory = new_trajectory;
  
}

void Dmp::analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs, Eigen::MatrixXd& xds, Eigen::MatrixXd& forcing_terms, Eigen::MatrixXd& fa_outputs) const
{
  int n_time_steps = ts.size();
  assert(n_time_steps>0);
  
  // Usually, we expect xs and xds to be of size T X dim(), so we resize to that. However, if the
  // input matrices were of size dim() X T, we return the matrices of that size by doing a 
  // transposeInPlace at the end. That way, the user can also request dim() X T sized matrices.
  bool caller_expects_transposed = (xs.rows()==dim() && xs.cols()==n_time_steps);
  
  // INTEGRATE SYSTEMS ANALYTICALLY AS MUCH AS POSSIBLE

  // Integrate phase
  MatrixXd xs_phase;
  MatrixXd xds_phase;
  phase_system_->analyticalSolution(ts,xs_phase,xds_phase);
  
  // Compute gating term
  MatrixXd xs_gating;
  MatrixXd xds_gating;
  gating_system_->analyticalSolution(ts,xs_gating,xds_gating);

  // Compute the output of the function approximator
  fa_outputs.resize(ts.size(),dim_orig());
  fa_outputs.fill(0.0);
  //if (isTrained())
  computeFunctionApproximatorOutput(xs_phase, fa_outputs);

  // Gate the output to get the forcing term
  MatrixXd xs_gating_rep = xs_gating.replicate(1,fa_outputs.cols());
  forcing_terms = fa_outputs.array()*xs_gating_rep.array();
  
  // Scale the forcing term, if necessary
  if (forcing_term_scaling_==G_MINUS_Y0_SCALING)
  {
    MatrixXd g_minus_y0_rep = (attractor_state()-initial_state()).transpose().replicate(n_time_steps,1);
    forcing_terms = forcing_terms.array()*g_minus_y0_rep.array();
  }
  else if (forcing_term_scaling_==AMPLITUDE_SCALING)
  {
    MatrixXd trajectory_amplitudes_rep = trajectory_amplitudes_.transpose().replicate(n_time_steps,1);
    forcing_terms = forcing_terms.array()*trajectory_amplitudes_rep.array();
  }
  
  
  MatrixXd xs_goal, xds_goal;
  // Get current delayed goal
  if (goal_system_==NULL)
  {
    // If there is no dynamical system for the delayed goal, the goal is
    // simply the attractor state               
    xs_goal  = attractor_state().transpose().replicate(n_time_steps,1);
    // with zero change
    xds_goal = MatrixXd::Zero(n_time_steps,dim_orig());
  } 
  else
  {
    // Integrate goal system and get current goal state
    goal_system_->analyticalSolution(ts,xs_goal,xds_goal);
  }

  xs.resize(n_time_steps,dim());
  xds.resize(n_time_steps,dim());

  int T = n_time_steps;
    
  xs.GOALM(T) = xs_goal;     xds.GOALM(T) = xds_goal;
  xs.PHASEM(T) = xs_phase;   xds.PHASEM(T) = xds_phase;
  xs.GATINGM(T) = xs_gating; xds.GATINGM(T) = xds_gating;

  
  // THE REST CANNOT BE DONE ANALYTICALLY
  
  // Reset the dynamical system, and get the first state
  double damping = spring_system_->damping_coefficient();
  SpringDamperSystem localspring_system_(tau(),initial_state(),attractor_state(),damping);
  
  // Set first attractor state
  localspring_system_.set_attractor_state(xs_goal.row(0));
  
  // Start integrating spring damper system
  int dim_spring = localspring_system_.dim();
  VectorXd x_spring(dim_spring), xd_spring(dim_spring); // todo Why are these needed?
  int t0 = 0;
  localspring_system_.integrateStart(x_spring, xd_spring);
  xs.row(t0).SPRING  = x_spring;
  xds.row(t0).SPRING = xd_spring;

  // Add forcing term to the acceleration of the spring state  
  xds.SPRINGM_Z(1) = xds.SPRINGM_Z(1) + forcing_terms.row(t0)/tau();

  // Initialize perturber, if necessary
  if (analytical_solution_perturber_==NULL && perturbation_standard_deviation_>0.0)
  {
    boost::normal_distribution<> normal(0, perturbation_standard_deviation_);
    analytical_solution_perturber_ = new boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >(rng, normal);
  }
 
  
  for (int tt=1; tt<n_time_steps; tt++)
  {
    double dt = ts[tt]-ts[tt-1];
    
    // Euler integration
    xs.row(tt).SPRING  = xs.row(tt-1).SPRING + dt*xds.row(tt-1).SPRING;
  
    // Set the attractor state of the spring system
    localspring_system_.set_attractor_state(xs.row(tt).GOAL);

    // Integrate spring damper system
    localspring_system_.differentialEquation(xs.row(tt).SPRING, xd_spring);
    xds.row(tt).SPRING = xd_spring;
    
    // If necessary add a perturbation. May be useful for some off-line tests.
    RowVectorXd perturbation = RowVectorXd::Constant(dim_orig(),0.0);
    if (analytical_solution_perturber_!=NULL)
      for (int i_dim=0; i_dim<dim_orig(); i_dim++)
        // Sample perturbation from a normal Gaussian distribution
        perturbation(i_dim) = (*analytical_solution_perturber_)();
      
    // Add forcing term to the acceleration of the spring state
    xds.row(tt).SPRING_Z = xds.row(tt).SPRING_Z + forcing_terms.row(tt)/tau() + perturbation;
    // Compute y component from z
    xds.row(tt).SPRING_Y = xs.row(tt).SPRING_Z/tau();
    
  } 
  
  if (caller_expects_transposed)
  {
    xs.transposeInPlace();
    xds.transposeInPlace();
  }
}

void Dmp::computeFunctionApproximatorInputsAndTargets(const Trajectory& trajectory, VectorXd& fa_inputs_phase, MatrixXd& f_target) const
{
  int n_time_steps = trajectory.length();
  double dim_data = trajectory.dim();
  
  if (dim_orig()!=dim_data)
  {
    cout << "WARNING: Cannot train " << dim_orig() << "-D DMP with " << dim_data << "-D data. Doing nothing." << endl;
    return;
  }
  
  // Integrate analytically to get goal, gating and phase states
  MatrixXd xs_ana;
  MatrixXd xds_ana;
  
  // Before, we would make clone of the dmp, and integrate it with the tau, and initial/attractor
  // state of the trajectory. However, Thibaut needed to call this from outside the Dmp as well,
  // with the tau/states of the this object. Therefore, we no longer clone. 
  // Dmp* dmp_clone = static_cast<Dmp*>(this->clone());
  // dmp_clone->set_tau(trajectory.duration());
  // dmp_clone->set_initial_state(trajectory.initial_y());
  // dmp_clone->set_attractor_state(trajectory.final_y());
  // dmp_clone->analyticalSolution(trajectory.ts(),xs_ana,xds_ana);
  analyticalSolution(trajectory.ts(),xs_ana,xds_ana);
  MatrixXd xs_goal   = xs_ana.GOALM(n_time_steps);
  MatrixXd xs_gating = xs_ana.GATINGM(n_time_steps);
  MatrixXd xs_phase  = xs_ana.PHASEM(n_time_steps);
  
  fa_inputs_phase = xs_phase;
  
  // Get parameters from the spring-dampers system to compute inverse
  double damping_coefficient = spring_system_->damping_coefficient();
  double spring_constant     = spring_system_->spring_constant();
  double mass                = spring_system_->mass();
  if (mass!=1.0)
  {
    cout << "WARNING: Usually, spring-damper system of the DMP should have mass==1, but it is " << mass << endl;
  }

  // Compute inverse
  f_target = tau()*tau()*trajectory.ydds() + (spring_constant*(trajectory.ys()-xs_goal) + damping_coefficient*tau()*trajectory.yds())/mass;
  
  //Factor out gating term
  for (unsigned int dd=0; dd<function_approximators_.size(); dd++)
    f_target.col(dd) = f_target.col(dd).array()/xs_gating.array();
  
  // Factor out scaling
  if (forcing_term_scaling_==G_MINUS_Y0_SCALING)
  {
    MatrixXd g_minus_y0_rep = (attractor_state()-initial_state()).transpose().replicate(n_time_steps,1);
    f_target = f_target.array()/g_minus_y0_rep.array();
  }
  else if (forcing_term_scaling_==AMPLITUDE_SCALING)
  {
    MatrixXd trajectory_amplitudes_rep = trajectory_amplitudes_.transpose().replicate(n_time_steps,1);
    f_target = f_target.array()/trajectory_amplitudes_rep.array();
  }
 
}

void Dmp::train(const Trajectory& trajectory)
{
  train(trajectory,"");
}

void Dmp::train(const Trajectory& trajectory, std::string save_directory, bool overwrite)
{
  
  // Set tau, initial_state and attractor_state from the trajectory 
  set_tau(trajectory.duration());
  set_initial_state(trajectory.initial_y());
  set_attractor_state(trajectory.final_y());

  // This needs to be computed for (optional) scaling of the forcing term.
  // Needs to be done BEFORE computeFunctionApproximatorInputsAndTargets
  trajectory_amplitudes_ = trajectory.getRangePerDim();
  
  VectorXd fa_input_phase;
  MatrixXd f_target;
  computeFunctionApproximatorInputsAndTargets(trajectory, fa_input_phase, f_target);
  
  // Some checks before training function approximators
  assert(!function_approximators_.empty());
  
  for (unsigned int dd=0; dd<function_approximators_.size(); dd++)
  {
    // This is just boring stuff to figure out if and where to store the results of training
    string save_directory_dim;
    if (!save_directory.empty())
    {
      if (function_approximators_.size()==1)
        save_directory_dim = save_directory;
      else
        save_directory_dim = save_directory + "/dim" + to_string(dd);
    }
    
    // Actual training is happening here.
    VectorXd fa_target = f_target.col(dd);
    if (function_approximators_[dd]==NULL)
    {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "WARNING: function approximator cannot be trained because it is NULL." << endl;
    }
    else
    {
      if (function_approximators_[dd]->isTrained())
        function_approximators_[dd]->reTrain(fa_input_phase,fa_target,save_directory_dim,overwrite);
      else
        function_approximators_[dd]->train(fa_input_phase,fa_target,save_directory_dim,overwrite);
    }
  }
  
  if (!save_directory.empty())
  {
    int n_time_steps = 101;
    VectorXd ts = VectorXd::LinSpaced(n_time_steps,0,tau());
    Trajectory traj_reproduced;
    analyticalSolution(ts,traj_reproduced);
    
    trajectory.saveToFile(save_directory,"traj_demonstration.txt");
    traj_reproduced.saveToFile(save_directory,"traj_reproduced.txt");
  }
  
}

void Dmp::getSelectableParameters(set<string>& selectable_values_labels) const {
  assert(function_approximators_.size()>0);
  for (int dd=0; dd<dim_orig(); dd++)
  {
    if (function_approximators_[dd]!=NULL)
    {
      if (function_approximators_[dd]->isTrained())
      {
        set<string> cur_labels;
        function_approximators_[dd]->getSelectableParameters(cur_labels);
        selectable_values_labels.insert(cur_labels.begin(), cur_labels.end());
      }
    }
  }
  selectable_values_labels.insert("goal");
  
  //cout << "selected_values_labels=["; 
  //for (string label : selected_values_labels) 
  //  cout << label << " ";
  //cout << "]" << endl;
  
}

void Dmp::setSelectedParameters(const set<string>& selected_values_labels)
{
  assert(function_approximators_.size()>0);
  for (int dd=0; dd<dim_orig(); dd++)
    if (function_approximators_[dd]!=NULL)
      if (function_approximators_[dd]->isTrained())
        function_approximators_[dd]->setSelectedParameters(selected_values_labels);

  // Call superclass for initializations
  Parameterizable::setSelectedParameters(selected_values_labels);

  VectorXi lengths_per_dimension = VectorXi::Zero(dim_orig());
  for (int dd=0; dd<dim_orig(); dd++)
  {
    if (function_approximators_[dd]!=NULL)
      if (function_approximators_[dd]->isTrained())
        lengths_per_dimension[dd] = function_approximators_[dd]->getParameterVectorSelectedSize();
    
    if (selected_values_labels.find("goal")!=selected_values_labels.end())
      lengths_per_dimension[dd]++;
  }
  
  setVectorLengthsPerDimension(lengths_per_dimension);
      
}

void Dmp::getParameterVectorMask(const std::set<std::string> selected_values_labels, Eigen::VectorXi& selected_mask) const
{
  assert(function_approximators_.size()>0);
  for (int dd=0; dd<dim_orig(); dd++)
  {
    assert(function_approximators_[dd]!=NULL);
    assert(function_approximators_[dd]->isTrained());
  }

  selected_mask.resize(getParameterVectorAllSize());
  selected_mask.fill(0);
  
  const int TMP_GOAL_NUMBER = -1;
  int offset = 0;
  VectorXi cur_mask;
  for (int dd=0; dd<dim_orig(); dd++)
  {
    function_approximators_[dd]->getParameterVectorMask(selected_values_labels,cur_mask);

    // This makes sure that the indices for each function approximator are different    
    int mask_offset = selected_mask.maxCoeff(); 
    for (int ii=0; ii<cur_mask.size(); ii++)
      if (cur_mask[ii]!=0)
        cur_mask[ii] += mask_offset;
        
    selected_mask.segment(offset,cur_mask.size()) = cur_mask;
    offset += cur_mask.size();
    
    // Goal
    if (selected_values_labels.find("goal")!=selected_values_labels.end())
      selected_mask(offset) = TMP_GOAL_NUMBER;
    offset++;
    
  }
  assert(offset == getParameterVectorAllSize());
  
  // Replace TMP_GOAL_NUMBER with current max value
  int goal_number = selected_mask.maxCoeff() + 1; 
  for (int ii=0; ii<selected_mask.size(); ii++)
    if (selected_mask[ii]==TMP_GOAL_NUMBER)
      selected_mask[ii] = goal_number;
    
}

int Dmp::getParameterVectorAllSize(void) const
{
  int total_size = 0;
  for (unsigned int dd=0; dd<function_approximators_.size(); dd++)
    total_size += function_approximators_[dd]->getParameterVectorAllSize();
  
  // For the goal
  total_size += dim_orig();
  return total_size;
}


void Dmp::getParameterVectorAll(VectorXd& values) const
{
  values.resize(getParameterVectorAllSize());
  int offset = 0;
  VectorXd cur_values;
  VectorXd attractor = attractor_state();
  for (int dd=0; dd<dim_orig(); dd++)
  {
    function_approximators_[dd]->getParameterVectorAll(cur_values);
    values.segment(offset,cur_values.size()) = cur_values;
    offset += cur_values.size();

    values(offset) = attractor(dd);
    offset++;
  }
}

void Dmp::setParameterVectorAll(const VectorXd& values)
{
  assert(values.size()==getParameterVectorAllSize());
  int offset = 0;
  VectorXd cur_values;
  VectorXd attractor(dim_orig());
  for (int dd=0; dd<dim_orig(); dd++)
  {
    int n_parameters_required = function_approximators_[dd]->getParameterVectorAllSize();
    cur_values = values.segment(offset,n_parameters_required);
    function_approximators_[dd]->setParameterVectorAll(cur_values);
    offset += n_parameters_required;
    
    attractor(dd) = values(offset);
    offset += 1;
  }
  
  // Set the goal
  set_attractor_state(attractor); 
}



void Dmp::set_tau(double tau) {
  DynamicalSystem::set_tau(tau);

  // Set value in all relevant subsystems also  
  spring_system_->set_tau(tau);
  if (goal_system_!=NULL)
    goal_system_->set_tau(tau);
  phase_system_ ->set_tau(tau);
  gating_system_->set_tau(tau);
}

void Dmp::set_initial_state(const VectorXd& y_init) {
  DynamicalSystem::set_initial_state(y_init);
  
  // Set value in all relevant subsystems also  
  spring_system_->set_initial_state(y_init);
  if (goal_system_!=NULL)
    goal_system_->set_initial_state(y_init);
}    

void Dmp::set_attractor_state(const VectorXd& y_attr) {
  DynamicalSystem::set_attractor_state(y_attr);
  
  // Set value in all relevant subsystems also  
  if (goal_system_!=NULL)
    goal_system_->set_attractor_state(y_attr);

  // Do NOT do the following. The attractor state of the spring system is determined by the goal 
  // system
  // spring_system_->set_attractor_state(y_attr);

}    

void Dmp::set_perturbation_analytical_solution(double perturbation_standard_deviation)
{
  perturbation_standard_deviation_ = perturbation_standard_deviation;
  analytical_solution_perturber_ = NULL;
}


string Dmp::toString(void) const
{
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("Dmp");
}


template<class Archive>
void Dmp::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(DynamicalSystem);
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Parameterizable);
  
  ar & BOOST_SERIALIZATION_NVP(goal_system_);
  ar & BOOST_SERIALIZATION_NVP(spring_system_);
  ar & BOOST_SERIALIZATION_NVP(phase_system_);
  ar & BOOST_SERIALIZATION_NVP(gating_system_);
  ar & BOOST_SERIALIZATION_NVP(function_approximators_);
  
  ar & BOOST_SERIALIZATION_NVP(forcing_term_scaling_);
  ar & BOOST_SERIALIZATION_NVP(trajectory_amplitudes_);
  
  ar & BOOST_SERIALIZATION_NVP(perturbation_standard_deviation_);
}

}
