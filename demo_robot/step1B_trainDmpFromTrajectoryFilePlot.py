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

lib_path = os.path.abspath('../python')
sys.path.append(lib_path)
from functionapproximators.functionapproximators_plotting import * 
from dmp.dmp_plotting import * 

def legendWithoutDuplicates(ax):
    handles, labels = ax.get_legend_handles_labels()  
    lgd = dict(zip(labels, handles))
    ax.legend(lgd.values(), lgd.keys())
#    plt.legend()
    

if __name__=='__main__':
    # Call the executable with the directory to which results should be written
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="directory to read data from")
    args = parser.parse_args()
    
    directory = args.directory
    
    # Plot trajectories    
    golden_ratio = 1.618 # graphs have nice proportions with this ratio
    n_subplots = 3
    fig3 = plt.figure(3,figsize=(golden_ratio*n_subplots*3,3))
    axs = [ fig3.add_subplot(1,n_subplots,ii+1) for ii in range(n_subplots) ]
    
    traj_demonstration = numpy.loadtxt(directory+"/traj_demonstration.txt")
    lines = plotTrajectory(traj_demonstration,axs)
    plt.setp(lines, linestyle='-',  linewidth=4, color=(0.8,0.8,0.8), label='demonstration')
    
    traj_reproduced = numpy.loadtxt(directory+"/traj_reproduced.txt")
    lines = plotTrajectory(traj_reproduced,axs)
    plt.setp(lines, linestyle='--', linewidth=2, color=(0.0,0.0,0.5), label='reproduced')
    
    legendWithoutDuplicates(axs[-1])
    
    fig3.canvas.set_window_title('Comparison between demonstration and reproduced') 
    plt.tight_layout()
    

    # Plot results of integrating the DMP
    tau           = numpy.loadtxt(directory+'/tau.txt')
    ts_xs_xds     = numpy.loadtxt(directory+'/reproduced_ts_xs_xds.txt')
    forcing_terms = numpy.loadtxt(directory+'/reproduced_forcing_terms.txt')
    fa_output     = numpy.loadtxt(directory+'/reproduced_fa_output.txt')
    
    fig2 = plt.figure(2,figsize=(golden_ratio*5*3,3*3))
    plotDmp(ts_xs_xds,fig2,forcing_terms,fa_output,[],tau)
    fig2.canvas.set_window_title('DMP integration') 
    plt.tight_layout()


    # Plot the results of training the function approximation 
    n_dims = (traj_demonstration.shape[1]-1)//3 # -1 for time, /3 for x/xd/xdd 
    fig1 = plt.figure(1,figsize=(golden_ratio*n_dims*4,4))
    for i_dim in range(n_dims):
        ax = fig1.add_subplot(1, n_dims, i_dim+1)
        plotFunctionApproximatorTrainingFromDirectory(directory+"dim"+str(i_dim),ax)
    
    fig1.canvas.set_window_title('Results of function approximation') 
    plt.tight_layout()
    
    fig1.savefig(directory+'/function_approximation.png')
    fig2.savefig(directory+'/dmp_integration.png')
    fig3.savefig(directory+'/trajectory_comparison.png')
    
    plt.show()
