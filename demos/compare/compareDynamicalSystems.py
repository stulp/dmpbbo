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


import os
import sys
import subprocess

import matplotlib.pyplot as plt
import numpy as np

# Include scripts for plotting
lib_path = os.path.abspath("../../python/")
sys.path.append(lib_path)

from dynamicalsystems.dynamicalsystems_plotting import *
from dynamicalsystems.ExponentialSystem import ExponentialSystem
from dynamicalsystems.SigmoidSystem import SigmoidSystem
from dynamicalsystems.SpringDamperSystem import SpringDamperSystem
from dynamicalsystems.TimeSystem import TimeSystem
from to_jsonpickle import *

def executeBinary(executable_name,arguments,print_command=False):
    
    if (not os.path.isfile(executable_name)):
        print("")
        print("ERROR: Executable '"+executable+"' does not exist.")
        print("Please call 'make install' in the build directory first.")
        print("")
        sys.exit(-1);
        
    command = executable_name+" "+arguments
    if print_command:
        print(command)
    
    subprocess.call(command, shell=True)


def plotComparison(ts,xs,xds,xs_cpp,xds_cpp,fig):
    axs = [fig.add_subplot(2, 2, p + 1) for p in range(4)]
    
    #plt.rc("text", usetex=True)
    #plt.rc("font", family="serif")
    
    h_cpp = []
    h_pyt = []
    h_diff = []
    
    h_pyt.extend(axs[0].plot(ts,xs,label='Python'))
    h_cpp.extend(axs[0].plot(ts,xs_cpp,label='C++'))
    axs[0].set_ylabel('x')
    
    h_pyt.extend(axs[1].plot(ts,xds,label='Python'))
    h_cpp.extend(axs[1].plot(ts,xds_cpp,label='C++'))
    axs[1].set_ylabel('dx')
    
    h_diff.extend(axs[2].plot(ts,xs-xs_cpp,label='diff'))
    axs[2].set_ylabel('diff x')
    
    h_diff.extend(axs[3].plot(ts,xds-xds_cpp,label='diff'))
    axs[3].set_ylabel('diff xd')
    
    plt.setp(h_pyt, linestyle='-', linewidth=4, color=(0.8,0.8,0.8))
    plt.setp(h_cpp, linestyle='--', linewidth=2, color=(0.2,0.2,0.8))
    plt.setp(h_diff, linestyle='-', linewidth=1, color=(0.8,0.2,0.2))
    
    for ax in axs:
        ax.set_xlabel('$t$')
        ax.legend()
    
    pass

if __name__ == "__main__":
    
    directory = "/tmp/compareDynamicalSystems/"
    os.makedirs(directory,exist_ok=True)

    ###########################################################################
    # Create all systems and add them to a dictionary

    # ExponentialSystem
    tau = 0.6  # Time constant
    x_init = np.array([0.5, 1.0])
    x_attr = np.array([0.8, 0.1])
    alpha = 6.0  # Decay factor
    dyn_systems = {"Exponential": ExponentialSystem(tau, x_init, x_attr, alpha)}

    # TimeSystem
    dyn_systems["Time"] = TimeSystem(tau)

    # TimeSystem (but counting down instead of up)
    count_down = True
    dyn_systems["TimeCountDown"] = TimeSystem(tau, count_down)

    # SigmoidSystem
    max_rate = -10
    inflection_ratio = 0.8
    dyn_systems["Sigmoid"] = SigmoidSystem(tau, x_init, max_rate, inflection_ratio)

    # SpringDamperSystem
    alpha = 12.0
    dyn_systems["SpringDamper"] = SpringDamperSystem(tau, x_init, x_attr, alpha)

    ###########################################################################
    # Start integration of all systems

    # Settings for the integration of the system
    dt = 0.01  # Integration step duration
    integration_duration = 1.25 * tau  # Integrate for longer than the time constant
    n_time_steps = int(np.ceil(integration_duration / dt)) + 1
    # Generate a vector of times, i.e. 0.0, dt, 2*dt, 3*dt .... n_time_steps*dt=integration_duration
    ts = np.linspace(0.0, integration_duration, n_time_steps)
    np.savetxt(directory+'/ts.txt',ts)

    fig_count = 1
    for name, dyn_system in dyn_systems.items():
        
        # Save the dynamical system to a json file 
        filename_json = directory+"/"+name+".json"
        with open(filename_json, 'w') as out_file:
            out_file.write(to_jsonpickle(dyn_system))
    
        # Call the binary, which does analyticalSolution and integration in C++
        exec_name = "../../build_dir_realtime/demos/compare/compareDynamicalSystems"
        arguments = directory+" "+name 
        executeBinary(exec_name,arguments,True)

        # Analytical solution
        xs, xds = dyn_system.analyticalSolution(ts)
        xs_cpp = np.loadtxt(directory+'/xs_analytical.txt')
        xds_cpp = np.loadtxt(directory+'/xds_analytical.txt')
        fig1 = plt.figure(fig_count,figsize=(10, 10))
        plotComparison(ts,xs,xds,xs_cpp,xds_cpp,fig1)
        fig1.suptitle(name + "System - Analytical")
        
        # Euler integration
        xs.fill(0.0)
        xds.fill(0.0)
        xs[0, :], xds[0, :] = dyn_system.integrateStart()
        for ii in range(1, n_time_steps):
            xs[ii, :], xds[ii, :] = dyn_system.integrateStepEuler(dt, xs[ii - 1, :])
        xs_cpp = np.loadtxt(directory+'/xs_euler.txt')
        xds_cpp = np.loadtxt(directory+'/xds_euler.txt')
        fig2 = plt.figure(fig_count+1,figsize=(10, 10))
        plotComparison(ts,xs,xds,xs_cpp,xds_cpp,fig2)
        fig2.suptitle(name + "System - Euler")
    
        # Runge-Kutta integration
        xs.fill(0.0)
        xds.fill(0.0)
        xs[0, :], xds[0, :] = dyn_system.integrateStart()
        for ii in range(1, n_time_steps):
            xs[ii, :], xds[ii, :] = dyn_system.integrateStepRungeKutta(dt, xs[ii - 1, :])
        xs_cpp = np.loadtxt(directory+'/xs_rungekutta.txt')
        xds_cpp = np.loadtxt(directory+'/xds_rungekutta.txt')
        fig3 = plt.figure(fig_count+2,figsize=(10, 10))
        plotComparison(ts,xs,xds,xs_cpp,xds_cpp,fig3)
        fig3.suptitle(name + "System - Runge-Kutta")
        
        save_me = True
        if save_me:
            fig1.savefig(os.path.join(directory,name + "System_analytical.png"))
            fig2.savefig(os.path.join(directory,name + "System_euler.png"))
            fig2.savefig(os.path.join(directory,name + "System_rungekutta.png"))
        fig_count += 3
        ##fig1.clf()
        #fig2.clf()
        #fig3.clf()

    plt.show()
