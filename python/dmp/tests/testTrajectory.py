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

lib_path = os.path.abspath('../../../python/')
sys.path.append(lib_path)

from dmp.dmp_plotting import *
from dmp.Dmp import *
from dmp.Trajectory import *
from functionapproximators.FunctionApproximatorRBFN import *

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
    

    print("Plotting")
    
    fig = plt.figure(1)
    n_plots = 5
    y_lims = [[-0.5, 0.8], [-3, 4], [-60,60]]
    for ii in range(n_plots):
        
        as_times = True
        crop_times = None
        if ii==0:
            crop_times = [-tau,2*tau]
        if ii==1:
            crop_times = [-tau,0.8*tau]
        if ii==2:
            crop_times = [0.2*tau,0.6*tau]
        
        if ii==4:
            traj.startTimeAtZero()
            
            
        i_plot = ii*3+1
        axs = [ fig.add_subplot(n_plots,3,i_plot), fig.add_subplot(n_plots,3,i_plot+1), fig.add_subplot(n_plots,3,i_plot+2) ] 
    
        lines = plotTrajectory(traj.asMatrix(),axs)
        for ii in range(len(axs)):
            axs[ii].set_xlim([0,tau])
            axs[ii].set_ylim(y_lims[ii])
            if crop_times is not None:
                axs[ii].plot([crop_times[0],crop_times[0]],y_lims[ii],'-r')
                axs[ii].plot([crop_times[1],crop_times[1]],y_lims[ii],'-r')
    
            
        if crop_times is not None:
            traj.crop(crop_times[0],crop_times[1],as_times)
            
    
    plt.show()
