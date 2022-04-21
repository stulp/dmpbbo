# This file is part of DmpBbo, a set of libraries and programs for the 
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2022 Freek Stulp
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
import os, sys
import argparse

lib_path = os.path.abspath('../python/')
sys.path.append(lib_path)

from dmp.dmp_plotting import *
from dmp.Dmp import *
from dmp.Trajectory import *
from functionapproximators.FunctionApproximatorRBFN import *
from functionapproximators.FunctionApproximatorLWR import *
from to_jsonpickle import *

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_file", help="file to read trajectory from")
    parser.add_argument("output_directory", help="directory to write dmp and other results to")
    parser.add_argument("--n", help="max number of basis functions", type=int, default=15)
    parser.add_argument("--show", action='store_true', help="Show plots")
    args = parser.parse_args()


    os.makedirs(args.output_directory,exist_ok=True)

    ################################################
    # Read trajectory and train DMP with it.

    print(f"Reading trajectory from: {args.trajectory_file}\n")
    traj = Trajectory.readFromFile(args.trajectory_file)
    n_dims =  traj.dim()
    peak_to_peak = np.ptp(traj._ys,axis=0) # Range of data; used later on
    
    mean_absolute_errors = []
    n_bfs_list = list(range(3,args.n))
    for n_bfs in n_bfs_list:
        
        function_apps = [FunctionApproximatorRBFN(n_bfs,0.7) for i_dim in range(n_dims)]
        dmp = Dmp.from_traj(traj, function_apps, "Dmp", 'KULVICIUS_2012_JOINING')

        # These are the parameters that will be optimized.
        dmp.setSelectedParameters("weights")
        
        ################################################
        # Save DMP to file
        
        filename = os.path.join(args.output_directory,f'dmp_trained_{n_bfs}.json')
        print("Saving trained DMP to: "+filename)
        with open(filename, 'w') as out_file:
            out_file.write(to_jsonpickle(dmp))
    
        ################################################
        # Analytical solution to compute difference
        
        ts = traj._ts
        ( xs_ana, xds_ana, forcing_terms_ana, fa_outputs_ana) = dmp.analyticalSolution(ts)
        traj_reproduced_ana = dmp.statesAsTrajectory(ts,xs_ana,xds_ana)
    
        mae = np.mean(abs(traj._ys - traj_reproduced_ana._ys),axis=0)
        mean_absolute_errors.append(mae)
        print()
        print(f'               Number of basis functions: {n_bfs}')
        print(f'MAE between demonstration and reproduced: {mae}')
        print(f'                           Range of data: {peak_to_peak}')
        print()
        
        
        ################################################
        # Integrate DMP
    
        tau_exec = 1.3*traj.duration
        dt = 0.01
        n_time_steps = int(tau_exec/dt)
        ts = np.zeros([n_time_steps,1])
        xs_step = np.zeros([n_time_steps,dmp._dim_x])
        xds_step = np.zeros([n_time_steps,dmp._dim_x])
        
        (x,xd) = dmp.integrateStart()
        xs_step[0,:] = x;
        xds_step[0,:] = xd;
        for tt in range(1,n_time_steps):
            ts[tt] = dt*tt
            (xs_step[tt,:],xds_step[tt,:]) = dmp.integrateStep(dt,xs_step[tt-1,:]);
    
        traj_reproduced = dmp.statesAsTrajectory(ts,xs_step,xds_step)
    
        ################################################
        # Plot results
        
        fig1 = plt.figure(1)
        fig1.clf()
        ts_xs_xds = np.column_stack((ts,xs_step,xds_step))
        plotDmp(ts_xs_xds,fig1)
        fig1.canvas.set_window_title(f'Step-by-step integration (n_bfs={n_bfs})') 
        
        fig2 = plt.figure(2)
        fig2.clf()
        axs = [ fig2.add_subplot(n) for n in [131,132,133]] 
        
        lines = plotTrajectory(traj.asMatrix(),axs)
        plt.setp(lines, linestyle='-',  linewidth=4, color=(0.8,0.8,0.8), label='demonstration')
    
        lines = plotTrajectory(traj_reproduced.asMatrix(),axs)
        plt.setp(lines, linestyle='--', linewidth=2, color=(0.0,0.0,0.5), label='reproduced')
        
        plt.legend()
        fig2.canvas.set_window_title(f'Comparison between demonstration and reproduced  (n_bfs={n_bfs})') 
        
        filename = f'dmp_trained_{n_bfs}.png'
        fig1.savefig(os.path.join(args.output_directory,filename))
        filename = f'trajectory_comparison_{n_bfs}.png'
        fig2.savefig(os.path.join(args.output_directory,filename))
        
            
    # Plot the mean absolute error
    fig3 = plt.figure(3)
    ax = fig3.add_subplot(111)
    ax.plot(n_bfs_list,mean_absolute_errors)
    ax.set_xlabel('number of basis functions')
    ax.set_ylabel('mean absolute error between demonstration and reproduced')
    filename = 'mean_absolute_errors.png'
    fig3.savefig(os.path.join(args.output_directory,filename))
                
    if args.show:
        plt.show()
    
    
