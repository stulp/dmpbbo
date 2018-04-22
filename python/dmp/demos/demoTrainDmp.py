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
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys, subprocess

lib_path = os.path.abspath('../../src/dmp/plotting/')
sys.path.append(lib_path)
lib_path = os.path.abspath('../../src/dynamicalsystems/plotting/')
sys.path.append(lib_path)
lib_path = os.path.abspath('../functionapproximators/')
sys.path.append(lib_path)

from plotTrajectory import plotTrajectory
from plotDmp import plotDmp
from Dmp import Dmp
from Trajectory import Trajectory
from FunctionApproximatorLWR import FunctionApproximatorLWR

if __name__=='__main__':

    tau = 0.5
    n_dims = 2
    n_time_steps = 51

    y_init = np.linspace(0.0,0.7,n_dims)
    y_attr = np.linspace(0.4,0.5,n_dims)
    
    ts = np.linspace(0,tau,n_time_steps)
    y_yd_ydd_viapoint = np.array([-0.2,0.4, 0.0,0.0, 0,0])
    viapoint_time = 0.4*ts[-1]
    traj = Trajectory.generatePolynomialTrajectoryThroughViapoint(ts, y_init, y_yd_ydd_viapoint, viapoint_time, y_attr)
    

    #function_apps = [None]*n_dims
    function_apps = [ FunctionApproximatorLWR(10), FunctionApproximatorLWR(10)]
    dmp = Dmp(tau, y_init, y_attr, function_apps)
    
    dmp.train(traj)

    tau_exec = 0.7
    n_time_steps = 71
    ts = np.linspace(0,tau_exec,n_time_steps)
    
    ( xs_ana, xds_ana, forcing_terms_ana, fa_outputs_ana) = dmp.analyticalSolution(ts)

    dt = ts[1]
    xs_step = np.zeros([n_time_steps,dmp.dim_])
    xds_step = np.zeros([n_time_steps,dmp.dim_])
    
    (x,xd) = dmp.integrateStart()
    xs_step[0,:] = x;
    xds_step[0,:] = xd;
    for tt in range(1,n_time_steps):
        (xs_step[tt,:],xds_step[tt,:]) = dmp.integrateStep(dt,xs_step[tt-1,:]); 

    print("Plotting")
    
    fig = plt.figure(1)
    axs = [ fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133) ] 
    
    lines = plotTrajectory(traj.asMatrix(),axs)
    plt.setp(lines, linestyle='-',  linewidth=4, color=(0.8,0.8,0.8), label='demonstration')

    traj_reproduced = dmp.statesAsTrajectory(ts,xs_step,xds_step)
    lines = plotTrajectory(traj_reproduced.asMatrix(),axs)
    plt.setp(lines, linestyle='--', linewidth=2, color=(0.0,0.0,0.5), label='reproduced')
    
    plt.legend()
    fig.canvas.set_window_title('Comparison between demonstration and reproduced') 
    
    
    fig = plt.figure(2)
    xs_xds = np.column_stack((xs_ana,xds_ana,ts))
    plotDmp(xs_xds,fig,forcing_terms_ana,fa_outputs_ana)
    fig.canvas.set_window_title('Analytical integration') 
    
    fig = plt.figure(3)
    xs_xds = np.column_stack((xs_step,xds_step,ts))
    plotDmp(xs_xds,fig)
    fig.canvas.set_window_title('Step-by-step integration') 
    
    
    plt.show()
