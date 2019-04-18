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


## \file demoDmp.py
## \author Freek Stulp
## \brief  Visualizes results of demoDmp.cpp
## 
## \ingroup Demos
## \ingroup Dmps

import numpy
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys

lib_path = os.path.abspath('../../python')
sys.path.append(lib_path)
from functionapproximators.functionapproximators_plotting import * 
from dmp.dmp_plotting import * 

if __name__=='__main__':
    # Call the executable with the directory to which results should be written
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="directory to read data from")
    args = parser.parse_args()
    
    directory = args.directory
    
    print("Plotting")
    
    fig = plt.figure(1)
    axs = [ fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133) ] 
    
    traj_demonstration = numpy.loadtxt(directory+"/traj_demonstration.txt")
    lines = plotTrajectory(traj_demonstration,axs)
    plt.setp(lines, linestyle='-',  linewidth=4, color=(0.8,0.8,0.8), label='demonstration')
    
    traj_reproduced = numpy.loadtxt(directory+"/traj_reproduced.txt")
    lines = plotTrajectory(traj_reproduced,axs)
    plt.setp(lines, linestyle='--', linewidth=2, color=(0.0,0.0,0.5), label='reproduced')
    
    n_dims = (traj_demonstration.shape[1]-1)//3 # -1 for time, /3 for x/xd/xdd 
    
    fig = plt.figure(2,figsize=(n_dims*5,5))
    for i_dim in range(n_dims):
        ax = fig.add_subplot(1, n_dims, i_dim+1)
        plotFunctionApproximatorTrainingFromDirectory(directory+"dim"+str(i_dim),ax)
       
    #plt.legend()
    #fig.canvas.set_window_title('Comparison between demonstration and reproduced') 
    
    # Read data
    #ts_xs_xds     = numpy.loadtxt(directory+'/reproduced_ts_xs_xds.txt')
    #forcing_terms = numpy.loadtxt(directory+'/reproduced_forcing_terms.txt')
    #fa_output     = numpy.loadtxt(directory+'/reproduced_fa_output.txt')
    
    #fig = plt.figure(2)
    #plotDmp(ts_xs_xds,fig,forcing_terms,fa_output)
    #fig.canvas.set_window_title('Analytical integration') 
    
    #ts_xs_xds     = numpy.loadtxt(directory+'/reproduced_step_ts_xs_xds.txt')
    #fig = plt.figure(3)
    #plotDmp(ts_xs_xds,fig)
    #fig.canvas.set_window_title('Step-by-step integration') 
    
    
    plt.show()
