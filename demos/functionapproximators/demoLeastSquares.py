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
from functionapproximators.leastSquares import *



def runLeastSquares(n_dims,use_offset,ax,directory=None):
        
    # Parameters for least squares 
    weights = np.ones(n_samples)
    regularization = 0.1

    if (directory == None):
        # Call Python function

        beta = weightedLeastSquares(inputs,targets,weights,use_offset,regularization)
    
        outputs = linearPrediction(inputs,beta)
        
    else:
        
        # Call executable compiled from demoLeastSquares.cpp
        
        cur_dir = '%s/%dD' % (directory, n_dims)
        if use_offset:
            cur_dir += '_use_offset'
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        
        np.savetxt(cur_dir+"/inputs.txt",inputs)
        np.savetxt(cur_dir+"/targets.txt",targets)
        np.savetxt(cur_dir+"/weights.txt",weights)
        
        executable = "../../bin/demoLeastSquares"
        
        if (not os.path.isfile(executable)):
            print("")
            print("ERROR: Executable '"+executable+"' does not exist.")
            print("Please call 'make install' in the build directory first.")
            print("")
            sys.exit(-1);
        
        directory = "/tmp/demoLeastSquares/"
        
        # Call the executable with the directory to which results should be written
        command = executable+" "
        command += " {:.8f}".format(regularization)
        if (use_offset):
            command += ' 1'
        else:
            command += ' 0'
        command += " "+cur_dir
        print(command)
        subprocess.call(command, shell=True)
        
        beta = np.loadtxt(cur_dir+"/beta.txt")
        outputs = np.loadtxt(cur_dir+"/outputs.txt")
    
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

    
    # Prepare figures
    figs = {}
    figs['Python'] = plt.figure(1,figsize=(10,10))     
    figs['C++'] = plt.figure(2,figsize=(10,10)) 
    for language in figs:
        figs[language].canvas.set_window_title(language) 
    
    i_subplot = 220
    for n_dims in [1,2]:
        
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
    
        
        # Add some noise
        for dd in range(len(targets)):
            targets[dd] = targets[dd] + 0.25*np.random.rand()

        for use_offset in [False,True]:
            i_subplot += 1
            
            for language in ['Python','C++']:
                
                # Prepare axes
                if n_dims==1:
                    ax = figs[language].add_subplot(i_subplot)
                else:
                    ax = figs[language].add_subplot(i_subplot, projection='3d')
                
                directory = None
                if (language=='C++'):
                    directory = '/tmp/demoLeastSquares/'
                    
                runLeastSquares(n_dims,use_offset,ax,directory)
        

    plt.show()
