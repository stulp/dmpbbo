# This file is part of DmpBbo, a set of libraries and programs for the 
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2018 Freek Stulp
#
# DmpBbo is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# DmpBbo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
# 

import numpy as np
import sys
import os

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from dmp.Trajectory import Trajectory

from dynamicalsystems.DynamicalSystem import DynamicalSystem
from dynamicalsystems.ExponentialSystem import ExponentialSystem
from dynamicalsystems.SigmoidSystem import SigmoidSystem
from dynamicalsystems.TimeSystem import TimeSystem
from dynamicalsystems.SpringDamperSystem import SpringDamperSystem


class Dmp(DynamicalSystem):

    def __init__(self,  tau, y_init, y_attr, function_apps, name="Dmp"):
        
        super().__init__(1, tau, y_init, y_attr, name)
        
        dim_orig = self.dim_orig_

        self.goal_system_  = ExponentialSystem(tau,y_init,y_attr,15,'goal')
        self.gating_system_ = SigmoidSystem(tau,np.ones(1),-20,0.9*tau,'gating') 
        self.phase_system_  = TimeSystem(tau,False,'phase')
        alpha = 20.0
        self.spring_system_ = SpringDamperSystem(tau,y_init,y_attr,alpha)
        
        self.function_approximators_ = function_apps

        # Make room for the subsystems
        self.dim_ = 3*dim_orig+2
        
        self.SPRING    = np.arange(0*dim_orig+0, 0*dim_orig+0 +2*dim_orig)
        self.SPRING_Y  = np.arange(0*dim_orig+0, 0*dim_orig+0 +dim_orig)
        self.SPRING_Z  = np.arange(1*dim_orig+0, 1*dim_orig+0 +dim_orig)
        self.GOAL      = np.arange(2*dim_orig+0, 2*dim_orig+0 +dim_orig)
        self.PHASE     = np.arange(3*dim_orig+0, 3*dim_orig+0 +       1)
        self.GATING    = np.arange(3*dim_orig+1, 3*dim_orig+1 +       1)
        #print(self.SPRING)
        #print(self.SPRING_Y)
        #print(self.SPRING_Z)
        #print(self.GOAL)
        #print(self.PHASE)
        #print(self.GATING)

        
    def set_tau(self,tau):
        
        self.tau_ = tau

        # Set value in all relevant subsystems also  
        self.spring_system_.set_tau(tau)
        if self.goal_system_:
            self.goal_system_.set_tau(tau)
        self.phase_system_ .set_tau(tau)
        self.gating_system_.set_tau(tau)
        
    def integrateStart(self):
        
        x = np.zeros(self.dim_)
        xd = np.zeros(self.dim_)
  
        # Start integrating goal system if it exists
        if self.goal_system_ is None:
            # No goal system, simply set goal state to attractor state
            x[self.GOAL] = self.attractor_state_
            xd[self.GOAL] = 0.0
        else:
            # Goal system exists. Start integrating it.
            (x[self.GOAL],xd[self.GOAL]) = self.goal_system_.integrateStart()
    
        # Set the attractor state of the spring system
        self.spring_system_.set_attractor_state(x[self.GOAL])
  
        # Start integrating all futher subsystems
        (x[self.SPRING],xd[self.SPRING]) = self.spring_system_.integrateStart()
        (x[self.PHASE ],xd[self.PHASE ]) = self.phase_system_.integrateStart()
        (x[self.GATING],xd[self.GATING]) = self.gating_system_.integrateStart()

        # Add rates of change
        xd = self.differentialEquation(x)
        return (x,xd)

    def differentialEquation(self,x):
        n_dims = self.dim_
        
        xd = np.zeros(x.shape)
        
        if self.goal_system_ is None:
            # If there is no dynamical system for the delayed goal, the goal is
            # simply the attractor state
            self.spring_system_.set_attractor_state(self.attractor_state_)
            # with zero change
            xd_goal = np.zeros(n_dims)
        else:
            # Integrate goal system and get current goal state
            self.goal_system_.set_attractor_state(self.attractor_state_)
            x_goal = x[self.GOAL]
            xd[self.GOAL] = self.goal_system_.differentialEquation(x_goal)
            # The goal state is the attractor state of the spring-damper system
            self.spring_system_.set_attractor_state(x_goal)
    
  
        # Integrate spring damper system
        #Forcing term is added to spring_state later
        xd[self.SPRING] = self.spring_system_.differentialEquation(x[self.SPRING])

  
        # Non-linear forcing term phase and gating systems
        xd[self.PHASE] = self.phase_system_.differentialEquation(x[self.PHASE])
        xd[self.GATING] = self.gating_system_.differentialEquation(x[self.GATING])

        fa_output = self.computeFunctionApproximatorOutput(x[self.PHASE]) 

        # Gate the output of the function approximators
        gating = x[self.GATING]
        forcing_term = gating*fa_output
        
  
        #// Scale the forcing term, if necessary
        #if (forcing_term_scaling_==G_MINUS_Y0_SCALING)
        #{
        #initial_state(initial_state_prealloc_)  
        #g_minus_y0_prealloc_ = (attractor_state_prealloc_-initial_state_prealloc_).transpose()
        #forcing_term_prealloc_ = forcing_term_prealloc_.array()*g_minus_y0_prealloc_.array()
        #}
        #else if (forcing_term_scaling_==AMPLITUDE_SCALING)
        #{
        #forcing_term_prealloc_ = forcing_term_prealloc_.array()*trajectory_amplitudes_.array()
        #}

        # Add forcing term to the ZD component of the spring state
        xd[self.SPRING_Z] += np.squeeze(forcing_term)/self.tau_
        
        return xd


    def computeFunctionApproximatorOutput(self,phase_state):
        n_time_steps = phase_state.size
        n_dims = self.dim_orig_
        fa_output = np.zeros([n_time_steps,n_dims])
        for i_fa in range(n_dims):
            if self.function_approximators_[i_fa]:
                if self.function_approximators_[i_fa].isTrained():
                    fa_output[:,i_fa] = self.function_approximators_[i_fa].predict(phase_state)
        return fa_output
        
    def analyticalSolution(self,ts):

        n_time_steps = ts.size
        
        # INTEGRATE SYSTEMS ANALYTICALLY AS MUCH AS POSSIBLE

        # Integrate phase
        ( xs_phase, xds_phase) = self.phase_system_.analyticalSolution(ts)
        
        # Compute gating term
        ( xs_gating, xds_gating ) = self.gating_system_.analyticalSolution(ts)
        
        # Compute the output of the function approximator
        fa_outputs = self.computeFunctionApproximatorOutput(xs_phase)

        # Gate the output to get the forcing term
        forcing_terms = fa_outputs*xs_gating
  
        # Scale the forcing term, if necessary
        #if (forcing_term_scaling_==G_MINUS_Y0_SCALING)
        #{
        #MatrixXd g_minus_y0_rep = (attractor_state()-initial_state()).transpose().replicate(n_time_steps,1)
        #forcing_terms = forcing_terms.array()*g_minus_y0_rep.array()
        #}
        #else if (forcing_term_scaling_==AMPLITUDE_SCALING)
        #{
        #MatrixXd trajectory_amplitudes_rep = trajectory_amplitudes_.transpose().replicate(n_time_steps,1)
        #forcing_terms = forcing_terms.array()*trajectory_amplitudes_rep.array()
        #}
  
  
        # Get current delayed goal
        if self.goal_system_ is None:
            # If there is no dynamical system for the delayed goal, the goal is
            # simply the attractor state               
            xs_goal  = np.tile(self.attractor_state_,(n_time_steps,1))
            # with zero change
            xds_goal = np.zeros(xs_goal.shape)
        else:
            # Integrate goal system and get current goal state
            (xs_goal,xds_goal) = self.goal_system_.analyticalSolution(ts)
            
            
        xs = np.zeros([n_time_steps,self.dim_])
        xds = np.zeros([n_time_steps,self.dim_])
    
        xs[:,self.GOAL] = xs_goal     
        xds[:,self.GOAL] = xds_goal
        xs[:,self.PHASE] = xs_phase   
        xds[:,self.PHASE] = xds_phase
        xs[:,self.GATING] = xs_gating 
        xds[:,self.GATING] = xds_gating

  
        # THE REST CANNOT BE DONE ANALYTICALLY
  
        # Reset the dynamical system, and get the first state
        damping = self.spring_system_.damping_coefficient_
        localspring_system = SpringDamperSystem(self.tau_,self.initial_state_,self.attractor_state_,damping)
  
        # Set first attractor state
        localspring_system.set_attractor_state(xs_goal[0,:])
  
        # Start integrating spring damper system
        (x_spring, xd_spring) = localspring_system.integrateStart()
        

        # For convenience
        SPRING = self.SPRING
        SPRING_Y = self.SPRING_Y
        SPRING_Z = self.SPRING_Z
        
        t0 = 0
        xs[t0,SPRING]  = x_spring
        xds[t0,SPRING]  = xd_spring

        # Add forcing term to the acceleration of the spring state  
        xds[0,SPRING_Z] = xds[0,SPRING_Z] + forcing_terms[t0,:]/self.tau_
  
        for tt in range(1,n_time_steps): 
            dt = ts[tt]-ts[tt-1]
    
            # Euler integration
            xs[tt,SPRING]  = xs[tt-1,SPRING] + dt*xds[tt-1,SPRING]
  
            # Set the attractor state of the spring system
            localspring_system.set_attractor_state(xs[tt,self.GOAL])

            # Integrate spring damper system
            xds[tt,SPRING] = localspring_system.differentialEquation(xs[tt,SPRING])
    
             # If necessary add a perturbation. May be useful for some off-line tests.
            #RowVectorXd perturbation = RowVectorXd::Constant(dim_orig(),0.0)
            #if (analytical_solution_perturber_!=NULL)
            #  for (int i_dim=0 i_dim<dim_orig() i_dim++)
            #    // Sample perturbation from a normal Gaussian distribution
            #    perturbation(i_dim) = (*analytical_solution_perturber_)()
      
            # Add forcing term to the acceleration of the spring state
            xds[tt,SPRING_Z] = xds[tt,SPRING_Z] + forcing_terms[tt,:]/self.tau_ #+ perturbation
            # Compute y component from z
            xds[tt,SPRING_Y] = xs[tt,SPRING_Z]/self.tau_
            
        return ( xs, xds, forcing_terms, fa_outputs)
        
        
    def train(self,trajectory):
  
        # Set tau, initial_state and attractor_state from the trajectory 
        self.set_tau(trajectory.ts_[-1])
        self.set_initial_state(trajectory.ys_[0,:])
        self.set_attractor_state(trajectory.ys_[-1,:])

        # This needs to be computed for (optional) scaling of the forcing term.
        # Needs to be done BEFORE computeFunctionApproximatorInputsAndTargets
        # zzz trajectory_amplitudes_ = trajectory.getRangePerDim()
  
        (fa_input_phase, f_target) = self.computeFunctionApproximatorInputsAndTargets(trajectory)
  
        for dd in range(self.dim_orig_):

            fa_target = f_target[:,dd]
            self.function_approximators_[dd].train(fa_input_phase,fa_target)
            
    def computeFunctionApproximatorInputsAndTargets(self,trajectory):
        n_time_steps = trajectory.ts_.size
        dim_data = trajectory.dim_
        assert(self.dim_orig_==dim_data)

        (xs_ana,xds_ana,forcing_terms, fa_outputs) = self.analyticalSolution(trajectory.ts_)
        xs_goal   = xs_ana[:,self.GOAL]
        xs_gating = xs_ana[:,self.GATING]
        xs_phase  = xs_ana[:,self.PHASE]
        
        fa_inputs_phase = xs_phase
  
        # Get parameters from the spring-dampers system to compute inverse
        damping_coefficient = self.spring_system_.damping_coefficient_
        spring_constant     = self.spring_system_.spring_constant_
        mass                = self.spring_system_.mass_
        # Usually, spring-damper system of the DMP should have mass==1
        assert(mass==1.0)

        #Compute inverse
        tau = self.tau_
        f_target = tau*tau*trajectory.ydds_ + (spring_constant*(trajectory.ys_-xs_goal) + damping_coefficient*tau*trajectory.yds_)/mass
  
        # Factor out gating term
        for dd in range(self.dim_orig_):
            f_target[:,dd] = f_target[:,dd]/np.squeeze(xs_gating)
  
        #  // Factor out scaling
        #  if (forcing_term_scaling_==G_MINUS_Y0_SCALING)
        #  {
        #    MatrixXd g_minus_y0_rep = (attractor_state()-initial_state()).transpose().replicate(n_time_steps,1)
        #    f_target = f_target.array()/g_minus_y0_rep.array()
        #  }
        #  else if (forcing_term_scaling_==AMPLITUDE_SCALING)
        #  {
        #    MatrixXd trajectory_amplitudes_rep = trajectory_amplitudes_.transpose().replicate(n_time_steps,1)
        #    f_target = f_target.array()/trajectory_amplitudes_rep.array()
        #  }
 
        return  (fa_inputs_phase, f_target)

    def statesAsTrajectory(self,ts, x_in, xd_in):
      
        # Left column is time
        return Trajectory(ts,x_in[:,self.SPRING_Y], xd_in[:,self.SPRING_Y], xd_in[:,self.SPRING_Z]/self.tau_)
  
