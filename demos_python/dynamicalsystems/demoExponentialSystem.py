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


import matplotlib.pyplot as plt
import numpy as np
import os, sys

# Include scripts for plotting
lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from dynamicalsystems.dynamicalsystems_plotting import * 

from dynamicalsystems.ExponentialSystem import ExponentialSystem

if __name__=='__main__':
    # Settings for the exponential system
    tau = 0.6  # Time constant
    initial_state = np.array([0.5, 1.0])
    attractor_state = np.array([0.8, 0.1])
    alpha = 6.0 # Decay rate

    # Construct the system
    system = ExponentialSystem(tau, initial_state, attractor_state, alpha)
  

    # Settings for the integration of the system
    dt = 0.004 # Integration step duration
    integration_duration = 1.5*tau # Integrate for longer than the time constant
    n_time_steps = int(np.ceil(integration_duration/dt))+1 # Number of time steps for the integration
    # Generate a vector of times, i.e. 0.0, dt, 2*dt, 3*dt .... n_time_steps*dt=integration_duration
    ts = np.linspace(0.0,integration_duration,n_time_steps)

    # NUMERICAL INTEGRATION 
    n_dims = system.dim_ # Dimensionality of the system
    xs_num = np.empty([n_dims,n_time_steps])
    xds_num = np.empty([n_dims,n_time_steps])

    # Use DynamicalSystemSystem::integrateStart to get the initial x and xd
    (xs_num[:,0],xds_num[:,0]) = system.integrateStart()
  
    # Use DynamicalSystemSystem::integrateStep to integrate numerically step-by-step
    for ii in range(1,n_time_steps):
        (xs_num[:,ii],xds_num[:,ii]) = system.integrateStep(dt,xs_num[:,ii-1])
        
    (xs_ana,xds_ana) = system.analyticalSolution(ts)
    
    # Plotting
    fig = plt.figure(1)
    data_ana = np.concatenate((xs_ana,xds_ana,np.atleast_2d(ts).T),axis=1)
    plotDynamicalSystem(data_ana,[fig.add_subplot(1,2,1), fig.add_subplot(1,2,2)])
    plt.title('analytical')
    
    fig = plt.figure(2)
    data_num = np.concatenate((xs_num.T,xds_num.T,np.atleast_2d(ts).T),axis=1)
    plotDynamicalSystem(data_num,[fig.add_subplot(1,2,1), fig.add_subplot(1,2,2)])
    plt.title('numerical')
    
    fig = plt.figure(3)
    axs      =  [fig.add_subplot(2,2,1), fig.add_subplot(2,2,2)]
    axs_diff =  [fig.add_subplot(2,2,3), fig.add_subplot(2,2,4)]
    plotDynamicalSystemComparison(data_ana,data_num,'analytical','numerical',axs,axs_diff)
    axs[1].legend()
    
    plt.show()
    

