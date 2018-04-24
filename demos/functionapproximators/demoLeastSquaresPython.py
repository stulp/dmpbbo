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



if __name__=='__main__':
    """Run some training sessions and plot results."""

    i_figure = 1
    for n_dims in [1,2]:
        for use_offset in [False,True]:
            print('\n===========================================')
            print('n_dims='+str(n_dims)+'  use_offset='+str(use_offset))
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
                
            # Parameters for least squares 
            weights = np.ones(n_samples)
            regularization = 0.1
        
            beta = weightedLeastSquares(inputs,targets,weights,use_offset,regularization)
        
            print("  beta=")
            print(beta)
            
            outputs = linearPrediction(inputs,beta)
            
            fig = plt.figure(i_figure,figsize=(15,5))
            i_figure += 1
            if n_dims==1:
                ax = fig.add_subplot(1, 1, 1)
            else:
                ax = Axes3D(fig)
        
            plotDataResiduals(inputs,targets,outputs,ax)
            plotDataTargets(inputs,targets,ax)
          
    plt.show()


