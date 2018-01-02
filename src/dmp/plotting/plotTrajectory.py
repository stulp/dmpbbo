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


from mpl_toolkits.mplot3d import Axes3D
import numpy                                                                  
import matplotlib.pyplot as plt                                               
import sys

# 
def plotTrajectory(trajectory,axs):
    """Plot a trajectory"""
    n_dims = (len(trajectory[0])-1)//3 # -1 for time, /3 because contains y,yd,ydd
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
      
    return lines

def plotTrajectoryFromFile(filename,axs):
    """Read trajectory, and plot it."""
    # Read data
    trajectory   = numpy.loadtxt(filename)
    lines = plotTrajectory(trajectory,axs)
    return lines

if __name__=='__main__':
    """Pass a filename argument, read trajectory, and plot."""

    if (len(sys.argv)==2):
        filename = str(sys.argv[1])
    else:
        print('\nUsage: '+sys.argv[0]+' <filename>    (trajectory is read from file)\n')
        sys.exit()
        
    fig = plt.figure()                                                            
    axs = [ fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133) ] 
    lines = plotTrajectoryFromFile(filename,axs)
    plt.setp(lines, linewidth=2)
    plt.show()


