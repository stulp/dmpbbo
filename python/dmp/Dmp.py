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
import matplotlib.pyplot as plt

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from dmp.Trajectory import Trajectory

from functionapproximators.Parameterizable import Parameterizable

from dynamicalsystems.DynamicalSystem import DynamicalSystem
from dynamicalsystems.ExponentialSystem import ExponentialSystem
from dynamicalsystems.SigmoidSystem import SigmoidSystem
from dynamicalsystems.TimeSystem import TimeSystem
from dynamicalsystems.SpringDamperSystem import SpringDamperSystem

from to_jsonpickle import *

class Dmp(DynamicalSystem,Parameterizable):

    def __init__(self, 
        tau, 
        y_init, y_attr,
        function_approximators=None,
        sigmoid_max_rate=-20,
        forcing_term_scaling='NO_SCALING',
        alpha_spring_damper=20.0, 
        phase_system=None, gating_system=None, goal_system=None, 
        ):
        """Initialize a DMP with function approximators and subsystems 
        
        Args:
            tau           - Time constant
            y_init        - Initial state
            y_attr        - Attractor state
            function_approximators - Function approximators for the forcing term
            forcing_term_scaling - Which method to use for scaling the forcing term
                ( "NO_SCALING", "G_MINUS_Y0_SCALING", "AMPLITUDE_SCALING" )
            alpha_spring_damper - \f$\alpha\f$ in the spring-damper system of the dmp
            goal_system   - Dynamical system to compute delayed goal
            phase_system  - Dynamical system to compute the phase
            gating_system - Dynamical system to compute the gating term
        """
        
        dim_dmp = 3*y_init.size+2
        super().__init__(1, tau, y_init, dim_dmp)
        #def __init__(self, order, tau, y_init, n_dims_x=None):
        
        self._y_attr = y_attr
        
        self._function_approximators = function_approximators
        
        self._forcing_term_scaling = forcing_term_scaling
 
        self._spring_system = SpringDamperSystem(tau,y_init,y_attr,alpha_spring_damper)
        
        # Set defaults for subsystems if necessary
        if not phase_system:
            phase_system = TimeSystem(tau,False)
        if not gating_system:
            o = np.ones(1)
            gating_system = SigmoidSystem(tau,o,sigmoid_max_rate,0.85) 
        if goal_system:
            goal_system  = ExponentialSystem(tau,y_init,y_attr,15)
            
        self._phase_system = phase_system
        self._gating_system = gating_system
        self._goal_system = goal_system
        
        self.ts_train_ = None

        self.goal_selected = False

        d = self._dim_y
        self.SPRING    = np.arange(0*d+0, 0*d+0 +2*d)
        self.SPRING_Y  = np.arange(0*d+0, 0*d+0 +1*d)
        self.SPRING_Z  = np.arange(1*d+0, 1*d+0 +1*d)
        self.GOAL      = np.arange(2*d+0, 2*d+0 +1*d)
        self.PHASE     = np.arange(3*d+0, 3*d+0 +1)
        self.GATING    = np.arange(3*d+1, 3*d+1 +1)
        
    def dim_dmp(self):
        return self._dim_y
  
    @classmethod
    def from_traj(cls,
        trajectory,
        function_approximators,
        dmp_type='KULVICIUS_2012_JOINING',
        forcing_term_scaling='NO_SCALING'
        ):
        """Initialize a DMP by training it from a trajectory. 
        
        Args:
            trajectory    - the trajectory to train on
            function_approximators - Function approximators for the forcing term
            dmp_type      - Type of the Dmp
                ( "IJSPEERT_2002_MOVEMENT", "KULVICIUS_2012_JOINING", "COUNTDOWN_2013")
            forcing_term_scaling - Which method to use for scaling the forcing term
                ( "NO_SCALING", "G_MINUS_Y0_SCALING", "AMPLITUDE_SCALING" )
            phase_system  - Dynamical system to compute the phase
            gating_system - Dynamical system to compute the gating term
        """
        
        # Relevant variables from trajectory
        tau = trajectory.ts_[-1]
        y_init = trajectory.ys_[0,:]
        y_attr = trajectory.ys_[-1,:]
        
        # Initialize dynamical systems

        if dmp_type=='IJSPEERT_2002_MOVEMENT':
            goal_system   = None
            phase_system  = ExponentialSystem(tau,1,0,4)
            gating_system = ExponentialSystem(tau,1,0,4)
            
        elif dmp_type in ['KULVICIUS_2012_JOINING','COUNTDOWN_2013']:
            goal_system   = ExponentialSystem(tau,y_init,y_attr,15)
            sigmoid_max_rate = -20
            gating_system = SigmoidSystem(tau,1,sigmoid_max_rate,0.85)
            count_down = dmp_type=='COUNTDOWN_2013'
            phase_system  = TimeSystem(tau,count_down);

        alpha_spring_damper=20.0
        dmp = cls(
            tau, 
            y_init, y_attr,
            function_approximators,
            None,
            forcing_term_scaling,
            alpha_spring_damper, 
            phase_system, gating_system, goal_system)
        
        dmp.train(trajectory)
      
        return dmp
        
    def set_tau(self,new_tau):
        
        self._tau = new_tau

        # Set value in all relevant subsystems also  
        self._spring_system.tau = new_tau
        if self._goal_system:
            self._goal_system.tau = new_tau
        self._phase_system.tau = new_tau
        self._gating_system.tau = new_tau
        
    def integrateStart(self):
        
        x = np.zeros(self._dim_x)
        xd = np.zeros(self._dim_x)
  
        # Start integrating goal system if it exists
        if self._goal_system is None:
            # No goal system, simply set goal state to attractor state
            x[self.GOAL] = self._y_attr
            xd[self.GOAL] = 0.0
        else:
            # Goal system exists. Start integrating it.
            (x[self.GOAL],xd[self.GOAL]) = self._goal_system.integrateStart()
    
        # Set the attractor state of the spring system
        self._spring_system.y_attr = x[self.GOAL]
  
        # Start integrating all futher subsystems
        (x[self.SPRING],xd[self.SPRING]) = self._spring_system.integrateStart()
        (x[self.PHASE ],xd[self.PHASE ]) = self._phase_system.integrateStart()
        (x[self.GATING],xd[self.GATING]) = self._gating_system.integrateStart()

        # Add rates of change
        xd = self.differentialEquation(x)
        return (x,xd)

    def differentialEquation(self,x):
        """The differential equation which defines the system.
   
        It relates state values to rates of change of those state values
        
        Args:
            x - current state (column vector of size dim() X 1)
            
        Returns:
            Rate of change in state (column vector of size dim() X 1)
        """
        
        xd = np.zeros(x.shape)
        
        if self._goal_system is None:
            # If there is no dynamical system for the delayed goal, the goal is
            # simply the attractor state
            self._spring_system.y_attr = self._y_attr
            # with zero change
            xd_goal = np.zeros(self._dim_x)
        else:
            # Integrate goal system and get current goal state
            self._goal_system.y_attr = self._y_attr
            x_goal = x[self.GOAL]
            xd[self.GOAL] = self._goal_system.differentialEquation(x_goal)
            # The goal state is the attractor state of the spring-damper system
            self._spring_system.y_attr = x_goal
    
  
        # Integrate spring damper system
        #Forcing term is added to spring_state later
        xd[self.SPRING] = self._spring_system.differentialEquation(x[self.SPRING])

  
        # Non-linear forcing term phase and gating systems
        xd[self.PHASE] = self._phase_system.differentialEquation(x[self.PHASE])
        xd[self.GATING] = self._gating_system.differentialEquation(x[self.GATING])

        fa_output = self.computeFunctionApproximatorOutput(x[self.PHASE]) 

        # Gate the output of the function approximators
        gating = x[self.GATING]
        forcing_term = gating*fa_output
        
  
        # Scale the forcing term, if necessary
        if (self._forcing_term_scaling=="G_MINUS_Y0_SCALING"):
            g_minus_y0 = (self.attractor_state_-self.initial_state_)
            forcing_term = forcing_term*g_minus_y0
        
        elif (self._forcing_term_scaling=="AMPLITUDE_SCALING"):
            forcing_term = forcing_term*self.trajectory_amplitudes_

        # Add forcing term to the ZD component of the spring state
        xd[self.SPRING_Z] += np.squeeze(forcing_term)/self._tau
        
        return xd


    def computeFunctionApproximatorOutput(self,phase_state):
        """Compute the outputs of the function approximators.
        
        Args:
            phase_state The phase states for which the outputs are computed.
            
        Returns:
            The outputs of the function approximators.
        """
        n_time_steps = phase_state.size
        fa_output = np.zeros([n_time_steps,self.dim_dmp()])
        
        if not self._function_approximators:
            return fa_output # No function approximators, return zeros
            
        for i_fa in range(self.dim_dmp()):
            if self._function_approximators[i_fa]:
                if self._function_approximators[i_fa].isTrained():
                    fa_output[:,i_fa] = self._function_approximators[i_fa].predict(phase_state)
        return fa_output
        
    def analyticalSolution(self,ts=None):
        """Return analytical solution of the system at certain times

        Args:
            ts - A vector of times for which to compute the analytical solutions.
            If None is passed, the ts vector from the trajectory used to train the DMP is used.
        
        Returns:
            xs - Sequence of state vectors. T x D or D x T matrix, where T is the number of times (the length of 'ts'), and D the size of the state (i.e. dim())
            xds - Sequence of state vectors (rates of change). T x D or D x T matrix, where T is the number of times (the length of 'ts'), and D the size of the state (i.e. dim())
            
        The output xs and xds will be of size D x T \em only if the matrix x you pass as an argument of size D x T. In all other cases (i.e. including passing an empty matrix) the size of x will be T x D. This feature has been added so that you may pass matrices of either size. 
        """
        if ts is None:
            if self.ts_train_ is None:
                print("Neither the argument 'ts' nor the member variable self.ts_train_ was set. Returning None.")
                return None
            else:
                # Set the times to the ones the Dmp was trained on.
                ts = self.ts_train_


        n_time_steps = ts.size
        
        # INTEGRATE SYSTEMS ANALYTICALLY AS MUCH AS POSSIBLE

        # Integrate phase
        ( xs_phase, xds_phase) = self._phase_system.analyticalSolution(ts)
        
        # Compute gating term
        ( xs_gating, xds_gating ) = self._gating_system.analyticalSolution(ts)
        
        # Compute the output of the function approximator
        fa_outputs = self.computeFunctionApproximatorOutput(xs_phase)

        # Gate the output to get the forcing term
        forcing_terms = fa_outputs*xs_gating
  
        # Scale the forcing term, if necessary
        if (self._forcing_term_scaling=="G_MINUS_Y0_SCALING"):
            g_minus_y0 = (self.attractor_state_-self.initial_state_)
            g_minus_y0_rep = np.tile(g_minus_y0,(n_time_steps,1))
            forcing_terms *= g_minus_y0_rep
            
        elif (self._forcing_term_scaling=="AMPLITUDE_SCALING"):
            trajectory_amplitudes_rep = np.tile(self.trajectory_amplitudes_,(n_time_steps,1))
            forcing_terms *= trajectory_amplitudes_rep
  
  
        # Get current delayed goal
        if self._goal_system is None:
            # If there is no dynamical system for the delayed goal, the goal is
            # simply the attractor state               
            xs_goal  = np.tile(self._y_attr,(n_time_steps,1))
            # with zero change
            xds_goal = np.zeros(xs_goal.shape)
        else:
            # Integrate goal system and get current goal state
            (xs_goal,xds_goal) = self._goal_system.analyticalSolution(ts)
            
            
        xs = np.zeros([n_time_steps,self._dim_x])
        xds = np.zeros([n_time_steps,self._dim_x])
    
        xs[:,self.GOAL] = xs_goal     
        xds[:,self.GOAL] = xds_goal
        xs[:,self.PHASE] = xs_phase   
        xds[:,self.PHASE] = xds_phase
        xs[:,self.GATING] = xs_gating 
        xds[:,self.GATING] = xds_gating

  
        # THE REST CANNOT BE DONE ANALYTICALLY
  
        # Reset the dynamical system, and get the first state
        damping = self._spring_system._damping_coefficient
        localspring_system = SpringDamperSystem(self._tau,self.y_init,self._y_attr,damping)
  
        # Set first attractor state
        localspring_system.y_attr = xs_goal[0,:]
  
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
        xds[0,SPRING_Z] = xds[0,SPRING_Z] + forcing_terms[t0,:]/self._tau
  
        for tt in range(1,n_time_steps): 
            dt = ts[tt]-ts[tt-1]
    
            # Euler integration
            xs[tt,SPRING]  = xs[tt-1,SPRING] + dt*xds[tt-1,SPRING]
  
            # Set the attractor state of the spring system
            localspring_system.y_attr = xs[tt,self.GOAL]

            # Integrate spring damper system
            xds[tt,SPRING] = localspring_system.differentialEquation(xs[tt,SPRING])
    
             # If necessary add a perturbation. May be useful for some off-line tests.
            #RowVectorXd perturbation = RowVectorXd::Constant(dim_orig(),0.0)
            #if (analytical_solution_perturber_!=NULL)
            #  for (int i_dim=0 i_dim<dim_orig() i_dim++)
            #    // Sample perturbation from a normal Gaussian distribution
            #    perturbation(i_dim) = (*analytical_solution_perturber_)()
      
            # Add forcing term to the acceleration of the spring state
            xds[tt,SPRING_Z] = xds[tt,SPRING_Z] + forcing_terms[tt,:]/self._tau #+ perturbation
            # Compute y component from z
            xds[tt,SPRING_Y] = xs[tt,SPRING_Z]/self._tau
            
        return ( xs, xds, forcing_terms, fa_outputs)
        
        
    def train(self,trajectory):
        """Train a DMP with a trajectory.
        
        Args:
            trajectory - The trajectory with which to train the DMP.
        """
        # Set tau, initial_state and attractor_state from the trajectory 
        self.set_tau(trajectory.ts_[-1])
        self.set_initial_state(trajectory.ys_[0,:])
        self.set_attractor_state(trajectory.ys_[-1,:])

        # This needs to be computed for (optional) scaling of the forcing term.
        # Needs to be done BEFORE computeFunctionApproximatorInputsAndTargets
        self.trajectory_amplitudes_ = trajectory.getRangePerDim()
  
        # Do not train function approximators if there are none
        if self._function_approximators:
            (fa_input_phase, f_target) = self.computeFunctionApproximatorInputsAndTargets(trajectory)

            for dd in range(self.dim_dmp()):
                fa_target = f_target[:,dd]
                self._function_approximators[dd].train(fa_input_phase,fa_target)
        
        # Save the times steps on which the Dmp was trained.
        # This is just a convenience function to be able to call 
        # analyticalSolution without the "ts" argument.
        self.ts_train_ = trajectory.ts_
            
    def computeFunctionApproximatorInputsAndTargets(self,trajectory):
        """Given a trajectory, compute the inputs and targets for the function approximators.
   
        For a standard Dmp the inputs will be the phase over time, and the targets will be the forcing term (with the gating function factored out).
        
        Args:
            trajectory - Trajectory, e.g. a demonstration.
            
        Returns:
            fa_inputs_phase - The inputs for the function approximators (phase signal)
            fa_targets - The targets for the function approximators (forcing term)
        """
        
        n_time_steps = trajectory.ts_.size
        dim_data = trajectory.dim_
        assert(self.dim_dmp()==dim_data)

        (xs_ana,xds_ana,forcing_terms, fa_outputs) = self.analyticalSolution(trajectory.ts_)
        xs_goal   = xs_ana[:,self.GOAL]
        xs_gating = xs_ana[:,self.GATING]
        xs_phase  = xs_ana[:,self.PHASE]
        
        fa_inputs_phase = xs_phase
  
        # Get parameters from the spring-dampers system to compute inverse
        damping_coefficient = self._spring_system._damping_coefficient
        spring_constant     = self._spring_system._spring_constant
        mass                = self._spring_system._mass
        # Usually, spring-damper system of the DMP should have mass==1
        assert(mass==1.0)

        #Compute inverse
        tau = self._tau
        f_target = tau*tau*trajectory.ydds_ + (spring_constant*(trajectory.ys_-xs_goal) + damping_coefficient*tau*trajectory.yds_)/mass
  
        # Factor out gating term
        for dd in range(self.dim_dmp()):
            f_target[:,dd] = f_target[:,dd]/np.squeeze(xs_gating)
  

        # Factor out scaling
        if (self._forcing_term_scaling=="G_MINUS_Y0_SCALING"):
            g_minus_y0 = (self.attractor_state_-self.initial_state_)
            g_minus_y0_rep = np.tile(g_minus_y0,(n_time_steps,1))
            f_target /= g_minus_y0_rep
            
        elif (self._forcing_term_scaling=="AMPLITUDE_SCALING"):
            trajectory_amplitudes_rep = np.tile(self.trajectory_amplitudes_,(n_time_steps,1))
            f_target /= trajectory_amplitudes_rep
 
        return  (fa_inputs_phase, f_target)

    def stateAsPosVelAcc(self, x_in, xd_in):
        return (x_in[self.SPRING_Y], xd_in[self.SPRING_Y], xd_in[self.SPRING_Z]/self._tau)
        
    def statesAsTrajectory(self,ts, x_in, xd_in):
        """Get the output of a DMP dynamical system as a trajectory.
        
        As it is a dynamical system, the state vector of a DMP contains the output of the goal, spring, phase and gating system. What we are most interested in is the output of the spring system. This function extracts that information, and also computes the accelerations of the spring system, which are only stored implicitely in xd_in because second order systems are converted to first order systems with expanded state.

        Args:
            ts    - A vector of times 
            x_in  - State vector over time
            xd_in - State vector over time (rates of change)
            
        Return:
            Trajectory representation of the DMP state vector output.
        """
        # Left column is time
        return Trajectory(ts,x_in[:,self.SPRING_Y], xd_in[:,self.SPRING_Y], xd_in[:,self.SPRING_Z]/self._tau)
  
    def set_initial_state(self,y_init_new):
        assert(y_init_new.size==self.dim_dmp())
        self._y_init = y_init_new
        
        # Set value in all relevant subsystems also  
        self._spring_system.y_init = y_init_new;
        if self._goal_system:
            self._goal_system.y_init = y_init_new
        
    def set_attractor_state(self,y_attr_new):
        assert(y_attr_new.size==self.dim_dmp())
        self._y_attr = y_attr_new
  
        # Set value in all relevant subsystems also  
        if self._goal_system:
            self._goal_system.y_attr = y_attr_new
        
        # Do NOT do the following. The attractor state of the spring system is 
        # determined by the goal system.
        # self._spring_system.y_attr = y_attr_new
        
    def getSelectableParameters(self):
        selectable = []
        for fa in self._function_approximators:
            selectable.extend(fa.getSelectableParameters())
        selectable.append('goal')
        # Remove duplicates
        return list(dict.fromkeys(selectable))

    def getSelectableParametersRecommended(self):
        """Return the names of the parameters that recommended to be selected.
        """
        for fa in self._function_approximators:
            selectable.extend(fa.getSelectableParameters())
        # Remove duplicates
        return list(dict.fromkeys(selectable))

    def setSelectedParameters(self,selected_values_labels):
        for fa in self._function_approximators:
            fa.setSelectedParameters(selected_values_labels)
        self.goal_selected = "goal" in selected_values_labels
        
    def getParameterVectorSelected(self):
        values = np.empty(0)
        for fa in self._function_approximators:
            if fa.isTrained():
                values = np.append(values,fa.getParameterVectorSelected())
        if self.goal_selected:
            values = np.append(values,self.attractor_state_)
        return values
        
    def setParameterVectorSelected(self,values):
        size = self.getParameterVectorSelectedSize()
        assert(len(values)==size)
        offset = 0
        for fa in self._function_approximators:
            if fa.isTrained():
                cur_size = fa.getParameterVectorSelectedSize()
                cur_values = values[offset:offset+cur_size]
                fa.setParameterVectorSelected(cur_values)                
                offset += cur_size
        if self.goal_selected:
            self.set_attractor_state(values[offset:offset+self.dim_orig_])
            
    def getParameterVectorSelectedSize(self):
        size = 0
        for fa in self._function_approximators:
            if fa.isTrained():
                size += fa.getParameterVectorSelectedSize()
        if self.goal_selected:
            size += self.dim_orig_
        return size

    def __str__(self):    
        return to_jsonpickle(self)


    @staticmethod
    def getDmpAxes(has_fa_output=False):
        n_cols = 5
        n_rows = 3 if has_fa_output else 2
        fig = plt.figure(figsize=(3*n_cols,3*n_rows))
        
        axs = [ fig.add_subplot(n_rows,5,i+1) for i in range(n_rows*5) ]
        return axs
    
    @staticmethod
    def plotStatic(tau, ts, xs, xds, **kwargs):
        forcing_terms = kwargs.get('forcing_terms',[]) 
        fa_output = kwargs.get('fa_output',[]) 
        ext_dims = kwargs.get('ext_dims',[]) 
        has_fa_output = len(forcing_terms)>0 or len(fa_output)>0
        
        axs = kwargs.get('axs') or Dmp.getDmpAxes(has_fa_output)
            
    
        # Dimensionality of dynamical system.
        dim_x = xs.shape[1]      
        # Dimensionality of the DMP. -2 because of phase and gating (which are 1D) and /3 because of spring system (which has dimensionality 2*n_dims_dmp) and goal system (which has dimensionality n_dims_dmp)
        n_dims_dmp = (dim_x-2)//3
        D = n_dims_dmp  # Abbreviation for convencience
    
        #define SPRING    segment(0*dim_orig()+0,2*dim_orig())
        #define SPRING_Y  segment(0*dim_orig()+0,dim_orig())
        #define SPRING_Z  segment(1*dim_orig()+0,dim_orig())
        #define GOAL      segment(2*dim_orig()+0,dim_orig())
        #define PHASE     segment(3*dim_orig()+0,       1)
        #define GATING    segment(3*dim_orig()+1,       1)
    
        # We will loop over each of the subsystems of the DMP: prepare some variables here
        # Names of each of the subsystems
        system_names   = ['phase','gating','goal','spring'];
        system_varname = [    'x',     'v',  '\mathbf{y}^{g_d}',  '\mathbf{y}' ];
        # The indices they have in the data 
        system_indices = [ range(3*D,3*D+1), range(3*D+1,3*D+2), range(2*D,3*D), range(0*D,2*D) ];
        system_order   = [       1,          1,           1,         2 ];
        # The subplot in which they are plotted (x is plotted here, xd in the subplot+1)
        subplot_offsets = [      1,          6,           3,         8  ];
        
        # Loop over each of the subsystems of the DMP
        n_systems = len(system_names)
        for i_system in range(n_systems):
          
            # Plot 'x' for this subsystem (analytical solution and step-by-step integration)
            #fig.suptitle(filename)
            cur_n_plots = 2
            if (system_order[i_system]==2):
                cur_n_plots = 3
            
            cur_axs = axs[subplot_offsets[i_system]-1:subplot_offsets[i_system]-1+cur_n_plots]
            cur_indices = list(system_indices[i_system])
            cur_xs = xs[:,cur_indices]
            cur_xds = xds[:,cur_indices]
            if (system_order[i_system]==2):
                lines = DynamicalSystem.plotStatic(tau,ts,cur_xs,cur_xds,axs=cur_axs,dim_y=n_dims_dmp);
            else:
                lines = DynamicalSystem.plotStatic(tau,ts,cur_xs,cur_xds,axs=cur_axs);
                
            if (system_names[i_system]=='gating'):
              plt.setp(lines,color='m')
              cur_axs[0].set_ylim([0, 1.1])
            if (system_names[i_system]=='phase'):
              cur_axs[0].set_ylim([0, 1.1])
              plt.setp(lines,color='c')
              
            for ii in range(len(cur_axs)):
              x = np.mean(cur_axs[ii].get_xlim())
              y = np.mean(cur_axs[ii].get_ylim())
              cur_axs[ii].text(x,y,system_names[i_system], horizontalalignment='center');
              if (ii==0):
                  cur_axs[ii].set_ylabel(r'$'+system_varname[i_system]+'$')
              if (ii==1):
                  cur_axs[ii].set_ylabel(r'$\dot{'+system_varname[i_system]+'}$')
              if (ii==2):
                  cur_axs[ii].set_ylabel(r'$\ddot{'+system_varname[i_system]+'}$')
            
        # todo Fix this
        if len(fa_output)>1:
            ax = axs[11-1]
            ax.plot(ts,fa_output)
            x = np.mean(ax.get_xlim())
            y = np.mean(ax.get_ylim())
            ax.text(x,y,'func. approx.', horizontalalignment='center');                                        
            ax.set_xlabel(r'time ($s$)');
            ax.set_ylabel(r'$f_\mathbf{\theta}('+system_varname[0]+')$');
        
        if len(forcing_terms)>1:
            ax = axs[12-1]
            ax.plot(ts,forcing_terms)
            x = np.mean(ax.get_xlim())
            y = np.mean(ax.get_ylim())
            ax.text(x,y,'forcing term', horizontalalignment='center');                                        
            ax.set_xlabel(r'time ($s$)');
            ax.set_ylabel(r'$v\cdot f_{\mathbf{\theta}}('+system_varname[0]+')$');
        
        if (len(ext_dims)>1):
            ax = axs[13-1]
            ax.plot(ts,ext_dims)
            x = np.mean(ax.get_xlim())
            y = np.mean(ax.get_ylim())
            ax.text(x,y,'extended dims', horizontalalignment='center');                                        
            ax.set_xlabel(r'time ($s$)');
            ax.set_ylabel(r'unknown');
    
        x_lim = [min(ts),max(ts)]
        for ax in plt.gcf().get_axes():
            ax.plot([tau,tau],ax.get_ylim(),'-k')
            ax.set_xlim(x_lim[0],x_lim[1])
