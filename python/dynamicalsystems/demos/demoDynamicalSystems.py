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
import os, sys, subprocess

# Include scripts for plotting
lib_path = os.path.abspath('../../src/dynamicalsystems/plotting') # zzz
sys.path.append(lib_path)
lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from plotDynamicalSystem import plotDynamicalSystem
from plotDynamicalSystemComparison import plotDynamicalSystemComparison
from ExponentialSystem import ExponentialSystem
from SigmoidSystem import SigmoidSystem
from TimeSystem import TimeSystem
from SpringDamperSystem import SpringDamperSystem


def runDynamicalSystemTest(dyn_system, demo_label):
  
    # Settings for the integration of the system
    dt = 0.004 # Integration step duration
    integration_duration = 1.5*dyn_system.tau_ # Integrate for longer than the time constant
    n_time_steps = int(np.ceil(integration_duration/dt))+1 # Number of time steps for the integration
    # Generate a vector of times, i.e. 0.0, dt, 2*dt, 3*dt .... n_time_steps*dt=integration_duration
    ts = np.linspace(0.0,integration_duration,n_time_steps)


    if demo_label == "tau":
        dyn_system.set_tau(0.5*dyn_system.tau_)
  
    if demo_label == "euler" or demo_label == "runge_kutta" or demo_label == "rungekutta":
        dyn_system.integration_method_ = demo_label.upper()

    if demo_label == "analytical":
      # ANALYTICAL SOLUTION 
      (xs_ana,xds_ana) = dyn_system.analyticalSolution(ts)
      return (xs_ana, xds_ana, ts)
      
      
    # NUMERICAL INTEGRATION 
    n_dims = dyn_system.dim_
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
            
        (xs_num[:,ii],xds_num[:,ii]) = dyn_system.integrateStep(dt,xs_num[:,ii-1])
  
    return (xs_num.T, xds_num.T, ts)


if __name__=='__main__':
    
    # See if input directory was passed
    if (len(sys.argv)<2 or len(sys.argv)>3):
        print('\nUsage: '+sys.argv[0]+' <test1> [test2]\n')
        print('Available test labels are:')
        print('   rungekutta - Use 4th-order Runge-Kutta numerical integration.')
        print('   euler      - Use simple Euler numerical integration.')
        print('   analytical - Compute analytical solution (rather than numerical integration)')
        print('   tau        - Change the time constant "tau"')
        print('   attractor  - Change the attractor state during the integration')
        print('   perturb    - Perturb the system during the integration')
        print('')
        print('If you call with two tests, the results of the two are compared in one plot.\n')
        sys.exit()
        
    demo_labels = []
    for arg in sys.argv[1:]:
      demo_labels.append(str(arg))
        
    
    # ExponentialSystem
    tau = 0.6 # Time constant
    initial_state = np.array([0.5, 1.0])
    attractor_state = np.array([0.8, 0.1])
    alpha = 6.0 # Decay factor
    dyn_systems = [ExponentialSystem(tau, initial_state, attractor_state, alpha)]
  
    # TimeSystem
    dyn_systems.append(TimeSystem(tau))

    # TimeSystem (but counting down instead of up)
    count_down = True
    dyn_systems.append(TimeSystem(tau,count_down,"TimeSystemCountDown"))
    
    # SigmoidSystem
    max_rate = -20
    inflection_point = tau*0.8
    dyn_systems.append(SigmoidSystem(tau, initial_state, max_rate, inflection_point))

    # SpringDamperSystem
    alpha = 12.0
    dyn_systems.append(SpringDamperSystem(tau, initial_state, attractor_state, alpha))

  
    # INTEGRATE ALL DYNAMICAL SYSTEMS IN THE ARRA
    
    # Loop through all systems, and do numerical integration and compute the analytical solution
    figure_number = 1
    for dyn_system in dyn_systems:
        name = dyn_system.name_
        print(name+": \t")
        all_data = []
        for demo_label in demo_labels:
            print(demo_label)
      
            # RUN THE CURRENT TEST FOR THE CURRENT SYSTEM
            (xs,xds,ts) = runDynamicalSystemTest(dyn_system, demo_label)
            
            cur_data = np.concatenate((xs,xds,np.atleast_2d(ts).T),axis=1)
            
            all_data.append(cur_data)
            
        # PLOTTING
        fig = plt.figure(figure_number)
        figure_number = figure_number+1
    
        if (len(demo_labels)==1):
            plotDynamicalSystem(all_data[0],[fig.add_subplot(1,2,1), fig.add_subplot(1,2,2)])
            fig.canvas.set_window_title(name+"  ("+demo_label+")") 
                
        else:
            data = all_data[0]
            data_compare = all_data[1] 
            
            axs      =  [fig.add_subplot(2,2,1), fig.add_subplot(2,2,2)]
            axs_diff =  [fig.add_subplot(2,2,3), fig.add_subplot(2,2,4)]
            # Bit of a hack... We happen to know that SpringDamperSystem is only second order system
            if (name == "SpringDamperSystem"):
              axs      =  [fig.add_subplot(2,3,1), fig.add_subplot(2,3,2), fig.add_subplot(2,3,3)]
              axs_diff =  [fig.add_subplot(2,3,4), fig.add_subplot(2,3,5), fig.add_subplot(2,3,6)]
              
            plotDynamicalSystemComparison(data,data_compare,demo_labels[0],demo_labels[1],axs,axs_diff)
            fig.canvas.set_window_title(name+"  ("+demo_labels[0]+" vs "+demo_labels[1]+")") 
            axs[1].legend()
        
    plt.show()

