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
    
    np.set_printoptions(precision=3,suppress=True)

    # Generate training data 
    n_samples_per_dim = 25
    inputs = np.linspace(0.0, 2.0,n_samples_per_dim)
    targets = 3*np.exp(-inputs)*np.sin(2*np.square(inputs))
    
    # Locally Weighted Regression
    intersection = 0.5;
    n_rfs = 9;
    fa = FunctionApproximatorLWR(n_rfs,intersection)
    
    fa.train(inputs,targets)
    outputs = fa.predict(inputs)
    lines = fa.getLines(inputs)
    activations = fa.getActivations(inputs)
    
    fig = plt.figure(1,figsize=(15,5))
    ax = fig.add_subplot(1, 1, 1)

    #plotLocallyWeightedLines(inputs,lines,ax,n_samples_per_dim,activations=None,activations_unnormalized=None):
    plotDataResiduals(inputs,targets,outputs,ax)
    plotDataTargets(inputs,targets,ax)
    plotLocallyWeightedLines(inputs,lines,ax,n_samples_per_dim,activations)
    
    plt.show()


