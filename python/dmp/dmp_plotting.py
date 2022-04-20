# This file is part of DmpBbo, a set of libraries and programs for the 
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
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


import sys, os
import numpy                                                                    
import matplotlib.pyplot as plt

# Include scripts for plotting
lib_path = os.path.abspath('../../python')
sys.path.append(lib_path)

from dynamicalsystems.DynamicalSystem import * 


def plotTrajectory(trajectory,axs,n_misc=0):
    """Plot a trajectory"""
    n_dims = (len(trajectory[0])-1-n_misc)//3 
    # -1 for time, /3 because contains y,yd,ydd
    time_index = 0;
    lines = axs[0].plot(trajectory[:,time_index],trajectory[:,1:n_dims+1], '-')
    axs[0].set_xlabel('time (s)');
    axs[0].set_ylabel('y');
    if (len(axs)>1):
      lines[len(lines):] = axs[1].plot(trajectory[:,time_index],trajectory[:,n_dims+1:2*n_dims+1], '-')
      axs[1].set_xlabel('time (s)');
      axs[1].set_ylabel('yd');
    if (len(axs)>2):
      lines[len(lines):] = axs[2].plot(trajectory[:,time_index],trajectory[:,2*n_dims+1:3*n_dims+1], '-')
      axs[2].set_xlabel('time (s)');
      axs[2].set_ylabel('ydd');
      
    if n_misc>0 and len(axs)>3:
      lines[len(lines):] = axs[3].plot(trajectory[:,time_index],trajectory[:,3*n_dims+1:], '-')
      axs[3].set_xlabel('time (s)');
      axs[3].set_ylabel('misc');
      
    x_lim = [min(trajectory[:,time_index]),max(trajectory[:,time_index])]
    for ax in axs:
        ax.set_xlim(x_lim[0],x_lim[1])
        
        
      
    return lines
