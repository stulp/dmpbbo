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
lib_path = os.path.abspath("../../python/")
sys.path.append(lib_path)

from functionapproximators.BasisFunction import *
from functionapproximators.FunctionApproximatorLWR import *
from functionapproximators.FunctionApproximatorRBFN import *


def targetFunction(n_samples_per_dim):

    n_dims = 1 if np.isscalar(n_samples_per_dim) else len(n_samples_per_dim)

    if n_dims == 1:
        inputs = np.linspace(0.0, 2.0, n_samples_per_dim)
        targets = 3 * np.exp(-inputs) * np.sin(2 * np.square(inputs))

    else:
        n_samples = np.prod(n_samples_per_dim)
        # Here comes naive inefficient implementation...
        x1s = np.linspace(-2.0, 2.0, n_samples_per_dim[0])
        x2s = np.linspace(-2.0, 2.0, n_samples_per_dim[1])
        inputs = np.zeros((n_samples, n_dims))
        targets = np.zeros(n_samples)
        ii = 0
        for x1 in x1s:
            for x2 in x2s:
                inputs[ii, 0] = x1
                inputs[ii, 1] = x2
                targets[ii] = 2.5 * x1 * np.exp(-np.square(x1) - np.square(x2))
                ii += 1

    return (inputs, targets)


def train(fa_name, n_dims):

    # Generate training data
    n_samples_per_dim = 30 if n_dims == 1 else [10, 10]
    (inputs, targets) = targetFunction(n_samples_per_dim)

    n_rfs = 9 if n_dims == 1 else [5, 5]  # Number of basis functions. To be used later.

    # Initialize function approximator
    if fa_name == "LWR":
        # This value for intersection is quite low. But for the demo it is nice
        # because it makes the linear segments quite obvious.
        intersection = 0.2
        fa = FunctionApproximatorLWR(n_rfs, intersection)
    else:
        intersection = 0.7
        fa = FunctionApproximatorRBFN(n_rfs, intersection)

    # Train function approximator with data
    fa.train(inputs, targets)

    # Make predictions for the targets
    outputs = fa.predict(inputs)

    if fa_name == "LWR":
        fa.setSelectedParamNames(["offsets","widths"])
    else:
        fa.setSelectedParamNames(["weights","widths"])
    values = fa.getParamVector()
    
    # Plotting
    inputs_min = np.min(inputs, axis=0)
    inputs_max = np.max(inputs, axis=0)
    w = 4 if n_dims==1 else 2
    a = 1 if n_dims==1 else 0.5
    
    fig = plt.figure(figsize=(10,5))
    if n_dims==1:
        axs = [ fig.add_subplot(121+i) for i in range(2) ]
    else:
        axs = [ fig.add_subplot(121+i,projection='3d') for i in range(2) ]
    
    for noise in ['additive','multiplicative']:
        ax = axs[0] if noise=='additive' else axs[1]
        
        
        # Original function
        fa.setParamVector(values)
        h, _ = fa.plotPredictionsGrid(inputs_min, inputs_max, ax=ax)
        plt.setp(h, color=[0.0, 0.0, 0.6], linewidth=w,alpha=a)
        if n_dims==1:
            hb, _ = fa.plotBasisFunctions(inputs_min, inputs_max, ax=ax)
            plt.setp(hb, color=[0.6, 0.0, 0.0], linewidth=w,alpha=a)
            
        
        # Perturbed function
        for i_sample in range(5):
            
            if noise=='additive':
                rand_vector = 0.05*np.random.standard_normal(values.shape)
                new_values = rand_vector+values
            else:
                rand_vector = 1.0 + 0.1*np.random.standard_normal(values.shape)
                new_values = rand_vector*values
            fa.setParamVector(new_values)
            
            if n_dims==1:
                hb, _ = fa.plotBasisFunctions(inputs_min, inputs_max, ax=ax)
                plt.setp(hb, color=[1.0, 0.5, 0.5], linewidth=w/3)
            h, _ = fa.plotPredictionsGrid(inputs_min, inputs_max, ax=ax)
            plt.setp(h,  color=[0.5, 0.5, 1.0], linewidth=w/2)
            
    
        ax.set_title(f"{fa_name} {n_dims}D ({noise} noise)")
        plt.gcf().canvas.set_window_title(f"{fa_name} {n_dims}D")


if __name__ == "__main__":
    """Run some training sessions and plot results."""

    for fa_name in ["RBFN", "LWR"]:
        for n_dims in [1, 2]:
            train(fa_name, n_dims)

    plt.show()
