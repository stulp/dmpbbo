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
import os, sys
import argparse

lib_path = os.path.abspath('../')
sys.path.append(lib_path)
from executeBinary import executeBinary

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from functionapproximators.functionapproximators_plotting import *
from functionapproximators.leastSquares import *


def runLeastSquaresDemo(n_dims, use_offset, regularization, directory, figure_number):   
    
    # Generate the data
    if n_dims==1:
        n_samples = 25
        inputs = np.linspace(0.0,2.0,n_samples)
        targets = 2.0*inputs + 3.0
    else:
        n_samples_x = 5
        n_samples_y = 5
        x = np.linspace(0, 2, n_samples_x)
        y = np.linspace(0, 2, n_samples_y)
        xv, yv = np.meshgrid(x, y)
        inputs = np.column_stack((xv.flatten(), yv.flatten()))
        targets = 2.0*inputs[:,0] + 1.0*inputs[:,1] + 3.0
        n_samples = targets.size

    weights = np.ones(n_samples)
    
    # Add some noise
    for dd in range(len(targets)):
        targets[dd] = targets[dd] + 0.25*np.random.rand()

    # Save the data to file
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savetxt(directory+"/inputs.txt",inputs)
    np.savetxt(directory+"/targets.txt",targets)
    np.savetxt(directory+"/weights.txt",weights)
    
    # Call the executable with the directory to which results should be written
    arguments = " {:.8f}".format(regularization)
    arguments += " "+str(int(use_offset))
    arguments += " "+directory
    executeBinary("../../bin/demoLeastSquares",arguments)
    
    # Results from file
    beta = np.loadtxt(directory+"/beta.txt")
    outputs = np.loadtxt(directory+"/outputs.txt")
    
    # Prepare axes
    fig = plt.figure(figure_number,figsize=(10,10))     
    i_subplot = 111
    if n_dims==1:
        ax = fig.add_subplot(i_subplot)
    else:
        ax = fig.add_subplot(i_subplot, projection='3d')
        
    # Do plotting
    plotDataResiduals(inputs,targets,outputs,ax)
    plotDataTargets(inputs,targets,ax)

    # Set title
    beta = np.atleast_1d(beta)
    title = 'f(x) = '
    np.set_printoptions(precision=3,suppress=True)
    for i_dim in range(n_dims):
        title += "{:.3f}".format(beta[i_dim])+'*x_'+str(i_dim+1)+' + '
    if use_offset:
        title += "{:.3f}".format(beta[n_dims])
    else:
        title = title[:-2] # To remove dangling '+' character 
    ax.set_title(title)
    


if __name__=='__main__':
    """Run some training sessions and plot results."""

    parser = argparse.ArgumentParser()
    parser.set_defaults(use_offset=False)
    parser.add_argument("n_dims", type=int, help="dimensionality of input data (1 or 2)")
    parser.add_argument("--use_offset", action='store_true', help="whether to use an offset in the linear model")
    parser.add_argument("--directory", help="directory",default='/tmp/demoLeastSquares/')
    parser.add_argument("--figure_number", type=int, help="figure number",default=1)
    parser.add_argument("--regularization", type=float, help="regularization term",default=0.0)
    args = parser.parse_args()
    
    runLeastSquaresDemo(args.n_dims, args.use_offset, args.regularization, args.directory, args.figure_number)   
    
    plt.show()

