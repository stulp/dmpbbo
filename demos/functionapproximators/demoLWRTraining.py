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


from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt                                               
import os, sys, subprocess

# Include scripts for plotting
lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from functionapproximators.functionapproximators_plotting import *
from functionapproximators.FunctionApproximatorLWR import *


if __name__=='__main__':
    """Run some training sessions and plot results."""
    
    # Generate training data 
    n_samples_per_dim = 25
    inputs = np.linspace(0.0, 2.0,n_samples_per_dim)
    targets = 3*np.exp(-inputs)*np.sin(2*np.square(inputs))
    
    # Initialize function approximator
    intersection = 0.5;
    n_rfs = 9;
    fa = FunctionApproximatorLWR(n_rfs,intersection)
    
    # Train function approximator with data
    fa.train(inputs,targets)
    
    # Make predictions for the targets
    outputs = fa.predict(inputs)
    
    # Make predictions on a grid
    n_samples_grid = 200
    inputs_grid = np.linspace(0.0, 2.0,n_samples_grid)
    outputs_grid = fa.predict(inputs_grid)
    lines_grid = fa.getLines(inputs_grid)
    activations_grid = fa.getActivations(inputs_grid)
    
    # Plotting
    fig = plt.figure(1,figsize=(15,5))
    ax = fig.add_subplot(121)
    ax.set_title('Python')
    plotGridPredictions(inputs_grid,outputs_grid,ax,n_samples_grid)
    plotDataResiduals(inputs,targets,outputs,ax)
    plotDataTargets(inputs,targets,ax)
    plotLocallyWeightedLines(inputs_grid,lines_grid,ax,n_samples_grid,activations_grid)
    

    # Here comes the same, but then in C++
    # Here, we will call the executable compiled from demoLWRTraining.cpp
    
    #Save training data to file
    directory = "/tmp/demoLWRTraining/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savetxt(directory+"inputs.txt",inputs)
    np.savetxt(directory+"targets.txt",targets)
    
    executable = "../../bin/demoLWRTraining"
    
    if (not os.path.isfile(executable)):
        print("")
        print("ERROR: Executable '"+executable+"' does not exist.")
        print("Please call 'make install' in the build directory first.")
        print("")
        sys.exit(-1);
    
    # Call the executable with the directory to which results should be written
    command = executable+" "+directory
    print(command)
    subprocess.call(command, shell=True)
    
    outputs = np.loadtxt(directory+"outputs.txt")
    inputs_grid = np.loadtxt(directory+"inputs_grid.txt")
    outputs_grid = np.loadtxt(directory+"predictions_grid.txt")
    lines_grid = np.loadtxt(directory+"lines_grid.txt")
    activations_grid = np.loadtxt(directory+"activations_grid.txt")
    activations_unnormalized_grid = np.loadtxt(directory+"activations_unnormalized_grid.txt")
    
    n_samples_grid = outputs_grid.size
    ax = fig.add_subplot(122)
    ax.set_title('C++')
    plotGridPredictions(inputs_grid,outputs_grid,ax,n_samples_grid)
    plotDataResiduals(inputs,targets,outputs,ax)
    plotDataTargets(inputs,targets,ax)
    plotLocallyWeightedLines(inputs_grid,lines_grid,ax,n_samples_grid,activations_grid,activations_unnormalized_grid)
    
    plt.show()


