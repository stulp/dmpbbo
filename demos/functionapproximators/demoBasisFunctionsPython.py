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

# Include scripts for plotting
lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from functionapproximators.functionapproximators_plotting import *
from functionapproximators.BasisFunction import Gaussian


if __name__=='__main__':
    """Run some training sessions and plot results."""
    

    n_centers = 5
    n_samples = 51
    
    centers = np.linspace(0.0,2.0,n_centers)
    widths = 0.2*np.ones(n_centers)
    inputs = np.linspace(0.0,2.0,n_samples)
    normalized=True

    kernel_acts =  Gaussian.activations(centers, widths, inputs, normalized)
      
    
    fig = plt.figure(1,figsize=(15,5))
    ax = fig.add_subplot(1, 1, 1)

    plotBasisFunctions(inputs,kernel_acts,ax,n_samples)
          
    plt.show()


