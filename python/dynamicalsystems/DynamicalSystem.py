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

class DynamicalSystem:

    def __init__(self,  order, tau, initial_state, attractor_state, name):
        assert(order==1 or order==2)
        initial_state = np.atleast_1d(initial_state)
        attractor_state = np.atleast_1d(attractor_state)
        assert(initial_state.shape==attractor_state.shape)
        
        # For 1st order systems, the dimensionality of the state vector 'x' is
        # 'dim'. For 2nd order systems, the system is expanded to x = [y z],
        # where 'y' and 'z' are both of dimensionality 'dim'. Therefore dim(x)
        # is 2*dim
        self.dim_ = initial_state.size*order
        
        # The dimensionality of the system before a potential rewrite
        self.dim_orig_ = initial_state.size
        
        self.tau_ = tau
        self.initial_state_ = initial_state
        
        self.attractor_state_ = attractor_state
        
        self.name_ = name
        
        self.integration_method_=  "EULER"
        
    def differentialEquation(self,x):
        raise NotImplementedError('subclasses must override updateDistribution()!')
        
    def analyticalSolution(self,ts):
        # Default implementation: call differentialEquation
        n_time_steps = ts.size
        xs = np.zeros([n_time_steps,self.dim_])
        xds = np.zeros([n_time_steps,self.dim_])
    
        (xs[0,:], xds[0,:]) = self.integrateStart()
        for tt in range(1,n_time_steps):
            dt = ts[tt] - ts[tt-1]
            (xs[tt,:],xds[tt,:]) = self.integrateStep(dt,xs[tt-1,:]) 
           
        return (xs,xds)

    def integrateStart(self,x_init=None):
        if x_init is not None:
            self.set_initial_state(x_init)
            
        # Pad the end with zeros: Why? In the spring-damper system, the state
        # consists of x = [y z]. 
        # The initial state only applies to y. Therefore, we set x = [y 0] 
        x = np.zeros(self.dim_)
        x[0:self.dim_orig_] = self.initial_state_
        
        # Return value (rates of change)
        return (x,self.differentialEquation(x))
        
    def integrateStep(self,dt, x):
      assert(dt>0.0)
      assert(x.size==self.dim_)
      if (self.integration_method_.upper() == "RUNGE_KUTTA" or self.integration_method_.upper() == "RUNGEKUTTA"):
        return self.integrateStepRungeKutta(dt, x)
      else:
        return self.integrateStepEuler(  dt, x)
            
    def integrateStepEuler(self, dt, x):
        # simple Euler integration
        xd_updated = self.differentialEquation(x)
        x_updated  = x + dt*xd_updated
        return (x_updated,xd_updated)
        
    def  integrateStepRungeKutta(self, dt, x):
        # 4th order Runge-Kutta for a 1st order system
        # http://en.wikipedia.org/wiki/Runge-Kutta_method#The_Runge.E2.80.93Kutta_method
        
        k1 = self.differentialEquation(x)
        input_k2 = x + dt*0.5*k1
        k2 = self.differentialEquation(input_k2)
        input_k3 = x + dt*0.5*k2
        k3 = self.differentialEquation(input_k3)
        input_k4 = x + dt*k3
        k4 = self.differentialEquation(input_k4)
          
        x_updated = x + dt*(k1 + 2.0*(k2+k3) + k4)/6.0
        xd_updated = self.differentialEquation(x_updated)
        return (x_updated,xd_updated)

    def set_tau(self,tau):
        assert(tau>0.0)
        self.tau_ = tau
        
    def set_initial_state(self,initial_state):
        assert(initial_state.size==self.dim_orig_)
        self.initial_state_ = initial_state
        
    def set_attractor_state(self,attractor_state):
        assert(attractor_state.size==self.dim_orig_)
        self.attractor_state_ = attractor_state