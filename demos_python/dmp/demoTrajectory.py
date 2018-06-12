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

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from dmp.dmp_plotting import *
from dmp.Trajectory import *

if __name__=='__main__':

    ts = np.linspace(0,0.5,101)
    y_first = np.array([0.0, 1.0])    
    yd_first = np.array([0.0, 0.0])    
    ydd_first = np.array([0.0, 0.0])    
    y_last = np.array([0.4, 0.5])    
    yd_last = np.array([10.0, 0])    
    ydd_last = np.array([0, 0])    
    
    fig = plt.figure(1)
    axs = [ fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133) ] 
    traj = Trajectory.generateMinJerkTrajectory(ts, y_first, y_last);
    plotTrajectory(traj.asMatrix(),axs)
    fig.canvas.set_window_title('min-jerk trajectory') 

    fig = plt.figure(2)
    axs = [ fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133) ] 
    traj = Trajectory.generatePolynomialTrajectory(ts, y_first, yd_first, ydd_first, y_last, yd_last, ydd_last)    
    plotTrajectory(traj.asMatrix(),axs)
    fig.canvas.set_window_title('polynomial trajectory') 

    fig = plt.figure(3)
    axs = [ fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133) ] 
    y_yd_ydd_viapoint = np.array([-0.2,0.3, 0.0,0.0, 0,0])
    viapoint_time = 0.4*ts[-1]
    traj = Trajectory.generatePolynomialTrajectoryThroughViapoint(ts, y_first, y_yd_ydd_viapoint, viapoint_time, y_last)
    plotTrajectory(traj.asMatrix(),axs)
    axs[0].plot(viapoint_time,y_yd_ydd_viapoint[0],'or')
    axs[0].plot(viapoint_time,y_yd_ydd_viapoint[1],'or')
    fig.canvas.set_window_title('polynomial viapoint trajectory') 

    plt.show()
