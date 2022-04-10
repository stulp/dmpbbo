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
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys, subprocess
import argparse

lib_path = os.path.abspath('../../../python/')
sys.path.append(lib_path)

from dmp.dmp_plotting import *
from dmp.Dmp import *
from dmp.Trajectory import *
from functionapproximators.FunctionApproximatorRBFN import *
from functionapproximators.FunctionApproximatorLWR import *
from to_jsonpickle import *

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="directory to write intermediate results to")
    parser.add_argument("--show", action='store_true', help="show result plots")
    args = parser.parse_args()

    os.makedirs(args.directory,exist_ok=True)

    ################################################
    # Read trajectory and train DMP with it.

    trajectory_file = 'trajectory.txt'
    print(f"Reading trajectory from: {trajectory_file}\n")
    traj = Trajectory.readFromFile(trajectory_file)
    n_dims =  traj.dim()
    
    n_bfs = 10
    function_apps = [FunctionApproximatorRBFN(n_bfs,0.7) for i_dim in range(n_dims)]
    dmp = Dmp.from_traj(traj, function_apps, "Dmp", 'KULVICIUS_2012_JOINING')

    ################################################
    # Save DMP to file
    
    filename = os.path.join(args.directory,f'dmp.json')
    print("Saving trained DMP to: "+filename)
    with open(filename, 'w') as out_file:
        out_file.write(to_jsonpickle(dmp))

    ################################################
    # Analytical solution to compute difference
    
    ts = traj.ts_
    ( xs_ana, xds_ana, forcing_terms_ana, fa_outputs_ana) = dmp.analyticalSolution(ts)
    traj_reproduced_ana = dmp.statesAsTrajectory(ts,xs_ana,xds_ana)

    ################################################
    # Integrate DMP

    #tau_exec = 1.3*traj.duration()
    #dt = 0.01
    #n_time_steps = int(tau_exec/dt)
    #ts = np.zeros([n_time_steps,1])
    n_time_steps = len(ts)
    dt = np.mean(np.diff(ts))
    xs_step = np.zeros([n_time_steps,dmp.dim_])
    xds_step = np.zeros([n_time_steps,dmp.dim_])
    
    (x,xd) = dmp.integrateStart()
    xs_step[0,:] = x;
    xds_step[0,:] = xd;
    for tt in range(1,n_time_steps):
        #ts[tt] = dt*tt
        (xs_step[tt,:],xds_step[tt,:]) = dmp.integrateStep(dt,xs_step[tt-1,:]);

    traj_reproduced = dmp.statesAsTrajectory(ts,xs_step,xds_step)

    np.savetxt(os.path.join(args.directory,f'ts.txt'),ts)
    
    ################################################
    # Plot results
    
    fig1 = plt.figure(1)
    fig1.clf()
    ts_xs_xds = np.column_stack((ts,xs_step,xds_step))
    ts_xs_xds_ana = np.column_stack((ts,xs_ana,xds_ana))
    plotDmp(ts_xs_xds_ana,fig1,forcing_terms_ana, fa_outputs_ana)
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
    
    filename = f'dmp_trained.png'
    fig1.savefig(os.path.join(args.directory,filename))
    filename = f'compare_python_demonstration_reproduction.png'
    fig2.savefig(os.path.join(args.directory,filename))
    
    # Save DMP
    
    
    executable = "../../../build_dir_debug/src/dmp/tests/testDmpCompareCppPython"
    
    if (not os.path.isfile(executable)):
        print("")
        print("ERROR: Executable '"+executable+"' does not exist.")
        print("Please call 'make install' in the build directory first.")
        print("")
        sys.exit(-1);
    
    # Call the executable with the directory to which results should be written
    command = executable+" "+args.directory
    print(command)
    subprocess.call(command, shell=True)
            
    
    # Plot the mean absolute error
    #fig3 = plt.figure(3)
    #ax = fig3.add_subplot(111)
    #ax.plot(n_bfs_list,mean_absolute_errors)
    #ax.set_xlabel('number of basis functions')
    #ax.set_ylabel('mean absolute error between demonstration and reproduced')
    #filename = 'mean_absolute_errors.png'
    #fig3.savefig(os.path.join(args.directory,filename))
    
    cpp_ts_xs_xds = np.loadtxt(args.directory+"/cpp_ts_xs_xds.txt")
    cpp_forcing_terms = np.loadtxt(args.directory+"/cpp_forcing_terms.txt")
    cpp_fa_output = np.loadtxt(args.directory+"/cpp_fa_output.txt")
    
    fig3 = plt.figure(3)
    plotDmp(cpp_ts_xs_xds,fig3,cpp_forcing_terms,cpp_fa_output)
       
    if args.show:
        plt.show()
    
    

def bla():
    """Run some training sessions and plot results."""
    
    # Generate training data 
    n_samples_per_dim = 25
    inputs = np.linspace(0.0, 2.0,n_samples_per_dim)
    targets = 3*np.exp(-inputs)*np.sin(2*np.square(inputs))
    
    fa_names = ["RBFN","LWR"]
    for fa_index in range(len(fa_names)):
        fa_name = fa_names[fa_index]
        
        #############################################
        # PYTHON
        
        # Initialize function approximator
        if fa_name=="LWR":
            intersection = 0.5;
            n_rfs = 9;
            fa = FunctionApproximatorLWR(n_rfs,intersection)
        else:
            intersection = 0.7;
            n_rfs = 9;
            fa = FunctionApproximatorRBFN(n_rfs,intersection)
        
        # Train function approximator with data
        fa.train(inputs,targets)
        
        # Make predictions for the targets
        outputs = fa.predict(inputs)
        
        # Make predictions on a grid
        n_samples_grid = 201
        inputs_grid = np.linspace(0.0, 2.0,n_samples_grid)
        outputs_grid = fa.predict(inputs_grid)
        if fa_name=="LWR":
            lines_grid = fa.getLines(inputs_grid)
        activations_grid = fa.getActivations(inputs_grid)
        
        # Plotting
        fig = plt.figure(fa_index,figsize=(15,5))
        fig.canvas.set_window_title(fa_name) 
        ax = fig.add_subplot(131)
        ax.set_title('Python')
        plotGridPredictions(inputs_grid,outputs_grid,ax,n_samples_grid)
        plotDataResiduals(inputs,targets,outputs,ax)
        plotDataTargets(inputs,targets,ax)
        if fa_name=="LWR":
            plotLocallyWeightedLines(inputs_grid,lines_grid,ax,n_samples_grid,activations_grid)
        if fa_name=="RBFN":
            plotBasisFunctions(inputs_grid,activations_grid,ax,n_samples_grid)
        
    
        #############################################
        # C++
        
        # Here comes the same, but then in C++
        # Here, we will call the executable compiled from demoLWRTraining.cpp
        
        #Save training data to file
        directory = "/tmp/demoTrain"+fa_name+"/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.savetxt(directory+"inputs.txt",inputs)
        np.savetxt(directory+"targets.txt",targets)
        
        executable = "../../../build_dir_debug/src/functionapproximators/tests/testTrainingCompareCppPython"
        
        if (not os.path.isfile(executable)):
            print("")
            print("ERROR: Executable '"+executable+"' does not exist.")
            print("Please call 'make install' in the build directory first.")
            print("")
            sys.exit(-1);
        
        # Call the executable with the directory to which results should be written
        command = executable+" "+fa_name+" "+directory
        command += " "+str(intersection)+" "+str(n_rfs)
        print(command)
        subprocess.call(command, shell=True)
        
        outputs = np.loadtxt(directory+"outputs.txt")
        inputs_grid_cpp = np.loadtxt(directory+"inputs_grid.txt")
        outputs_grid_cpp = np.loadtxt(directory+"predictions_grid.txt")
        activations_grid = np.loadtxt(directory+"activations_grid.txt")
        
        n_samples_grid = outputs_grid.size
        ax = fig.add_subplot(132)
        ax.set_title('C++')
        plotGridPredictions(inputs_grid_cpp,outputs_grid_cpp,ax,n_samples_grid)
        plotDataResiduals(inputs,targets,outputs,ax)
        plotDataTargets(inputs,targets,ax)
        if fa_name=="LWR":
            lines_grid = np.loadtxt(directory+"lines_grid.txt")
            activations_unnormalized_grid = np.loadtxt(directory+"activations_unnormalized_grid.txt")
            plotLocallyWeightedLines(inputs_grid_cpp,lines_grid,ax,n_samples_grid,activations_grid,activations_unnormalized_grid)
        if fa_name=="RBFN":
            plotBasisFunctions(inputs_grid_cpp,activations_grid,ax,n_samples_grid)
    
    
    
        #############################################
        # Difference between Python and C++
        
        ax = fig.add_subplot(133)
        diff_exists = False
        # inputs must be of same length
        if (len(inputs_grid)==len(inputs_grid_cpp)):
            inputs_diff = inputs_grid-inputs_grid_cpp
            # and have the same values
            if (np.max(np.abs(inputs_diff))<0.0000000001*inputs.ptp()):
                diff_exists = True
                outputs_diff = outputs_grid-outputs_grid_cpp
                line_handle = ax.plot(inputs_grid,outputs_diff)
                plt.setp(line_handle, color='red')                  
                ax.set_title('diff')
        
        if not diff_exists:
            # Instead of plotting diff, plot both curves over one another
            plotDataTargets(inputs,targets,ax)
            line_handle = ax.plot(inputs_grid,outputs_grid)
            line_handle_cpp = ax.plot(inputs_grid_cpp,outputs_grid_cpp)
            plt.setp(line_handle, label='Python', color='blue')
            plt.setp(line_handle_cpp, label='C++', color='green')
            plt.legend()
        
                
        
    plt.show()


