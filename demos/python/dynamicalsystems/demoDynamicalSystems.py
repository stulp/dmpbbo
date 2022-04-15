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
lib_path = os.path.abspath('../../../python/')
sys.path.append(lib_path)

from dynamicalsystems.dynamicalsystems_plotting import * 

from dynamicalsystems.ExponentialSystem import ExponentialSystem
from dynamicalsystems.SigmoidSystem import SigmoidSystem
from dynamicalsystems.TimeSystem import TimeSystem
from dynamicalsystems.SpringDamperSystem import SpringDamperSystem

def runDynamicalSystemTest(dyn_system, demo_label):
  
    # Settings for the integration of the system
    dt = 0.01 # Integration step duration
    integration_duration = 1.5*dyn_system.tau_ # Integrate for longer than the time constant
    n_time_steps = int(np.ceil(integration_duration/dt))+1 # Number of time steps for the integration
    # Generate a vector of times, i.e. 0.0, dt, 2*dt, 3*dt .... n_time_steps*dt=integration_duration
    ts = np.linspace(0.0,integration_duration,n_time_steps)

                                       
    if demo_label == "tau":
        dyn_system.set_tau(0.5*dyn_system.tau_)

    if demo_label == "analytical":
      # ANALYTICAL SOLUTION 
      (xs_ana,xds_ana) = dyn_system.analyticalSolution(ts)
      return (ts, xs_ana, xds_ana)
      
      
    # NUMERICAL INTEGRATION 
    n_dims = dyn_system.dim_x_
    xs_num = np.empty([n_dims,n_time_steps])
    xds_num = np.empty([n_dims,n_time_steps])
    
    (xs_num[:,0],xds_num[:,0]) = dyn_system.integrateStart()
    
    for ii in range(1,n_time_steps):
        
        if demo_label == "attractor":
            if ii == int(np.ceil(0.3*n_time_steps)):
                dyn_system.set_attractor_state(-0.2+dyn_system.attractor_state_)
                
        if demo_label == "perturb":
            if ii == int(np.ceil(0.3*n_time_steps)):
                xs_num[:,ii-1] = xs_num[:,ii-1]-0.2
            
        if demo_label == "euler":
            (xs_num[:,ii],xds_num[:,ii]) = dyn_system.integrateStepEuler(dt,xs_num[:,ii-1])
        else:
            (xs_num[:,ii],xds_num[:,ii]) = dyn_system.integrateStepRungeKutta(dt,xs_num[:,ii-1])
  
    return (ts, xs_num.T, xds_num.T)


if __name__=='__main__':
    
    available_demo_labels = {
        'rungekutta' :'Use 4th-order Runge-Kutta numerical integration.',
        'euler'      :'Use simple Euler numerical integration.',
        'analytical' :'Compute analytical solution (rather than numerical integration)',
        'tau'        :'Change the time constant "tau"',
        'attractor'  :'Change the attractor state during the integration',
        'perturb'    :'Perturb the system during the integration',
    }

    if len(sys.argv)>1:
        demo_labels = sys.argv[1:]
    else:
        demo_labels = ['euler']
        print('\nUsage: '+sys.argv[0]+' <test1> [test2]\n')
        print('Available test labels are:')
        for label, explanation in available_demo_labels.items():    
            print('   '+label+' - '+explanation)
        print('')
        print('If you call with two tests, the results of the two are compared in one plot.\n')
        
        
        
    
    # ExponentialSystem
    tau = 0.6 # Time constant
    x_init = np.array([0.5, 1.0])
    x_attr = np.array([0.8, 0.1])
    alpha = 6.0 # Decay factor
    dyn_systems = {"ExponentialSystem": ExponentialSystem(tau, x_init, x_attr, alpha)}
  
    # TimeSystem
    dyn_systems["TimeSystem"] = TimeSystem(tau)
    
    # TimeSystem (but counting down instead of up)
    count_down = True
    dyn_systems["TimeSystemCountDown"] = TimeSystem(tau,count_down)
        
    # SigmoidSystem
    max_rate = -20
    inflection_point = tau*0.8
    dyn_systems["SigmoidSystem"] = SigmoidSystem(tau, x_init, max_rate, inflection_point)
    
    # SpringDamperSystem
    alpha = 12.0
    dyn_systems["SpringDamperSystem"] = SpringDamperSystem(tau, x_init, x_attr, alpha)
    
  
    # INTEGRATE ALL DYNAMICAL SYSTEMS IN THE ARRAY
    
    # Loop through all systems, and do numerical integration and compute the analytical solution
    figure_number = 1
    for name, dyn_system in dyn_systems.items():
        print(name+": \t")
        
        # PLOTTING
        fig = plt.figure(figure_number)
        figure_number = figure_number+1
        axs = [fig.add_subplot(1,3,1), fig.add_subplot(1,3,2), fig.add_subplot(1,3,3)]
        
        for demo_label in demo_labels:
            print("    "+demo_label)
            (ts,xs,xds) = runDynamicalSystemTest(dyn_system, demo_label)
            lines = dyn_system.plot(ts,xs,xds,axs)
            
            plt.setp(lines,label=demo_label)
            if demo_label==demo_labels[0]:
                plt.setp(lines,linestyle='-',  linewidth=4, color=(0.8,0.8,0.8))
            else:
                plt.setp(lines,linestyle='--', linewidth=2, color=(0.0,0.0,0.5))
                
        for ax in axs:
            ax.legend()
        #fig.savefig(f'{name}.png')

    plt.show()

