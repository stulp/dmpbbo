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
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.

import os, sys

lib_path = os.path.abspath('../../../python/')
sys.path.append(lib_path)

from functionapproximators.FunctionApproximator import FunctionApproximator
from functionapproximators.BasisFunction import *
from functionapproximators.leastSquares import *

class FunctionApproximatorLWR(FunctionApproximator):
    
    def __init__(self,n_basis_functions_per_dim, intersection_height=0.5, regularization=0.0):
        
        self._meta_params = {
            'n_basis_functions_per_dim': n_basis_functions_per_dim,
            'intersection_height': intersection_height,
            'regularization': regularization,
        }

        # Initialize model parameters with empty lists
        labels = ['centers','widths','slopes','offsets']
        self._model_params = {label: [] for label in labels}
        
        self._selected_values_labels = ['slopes','offsets']
        
    def train(self,inputs,targets):

        # Determine the centers and widths of the basis functions, given the input data range
        min_vals = inputs.min(axis=0)
        max_vals = inputs.max(axis=0)
        n_bfs_per_dim = self._meta_params['n_basis_functions_per_dim']
        height = self._meta_params['intersection_height']
        (centers,widths) = getCentersAndWidths(min_vals, max_vals, n_bfs_per_dim, height)
       
        # Get the activations of the basis functions 
        self._model_params['widths'] = widths
        self._model_params['centers'] = centers
        activations = self.getActivations(inputs)

        # Parameters for the weighted least squares regressions
        use_offset = True
        n_kernels = np.prod(n_bfs_per_dim)
        n_betas = 1
        if (use_offset):
            n_betas += 1
        betas = np.ones([n_kernels,n_betas])
        
        # Perform one weighted least squares regression for each kernel
        reg = self._meta_params['regularization']
        for i_kernel in range(n_kernels):
            weights = activations[:,i_kernel]
            beta = weightedLeastSquares(inputs,targets,weights,use_offset,reg)
            betas[i_kernel,:] = beta.T
    
        self._model_params['offsets'] = np.atleast_2d(betas[:,-1]).T
        self._model_params['slopes'] = np.atleast_2d(betas[:,0:-1])

    def getActivations(self,inputs):
        normalize_activations = True
        activations = Gaussian.activations(self._model_params['centers'],self._model_params['widths'],inputs,normalize_activations)
        return activations
        
    def getLines(self,inputs):
        if inputs.ndim==1:
            # Otherwise matrix multiplication below will not work
            inputs = np.atleast_2d(inputs).T
        
        slopes = self._model_params['slopes']
        offsets = self._model_params['offsets']
        
        # Compute the line segments
        n_lines = self._model_params['offsets'].size 
        n_samples = inputs.shape[0]
        lines = np.zeros([n_samples,n_lines])
        for i_line in range(n_lines):
            # Apparently, not everybody has python3.5 installed, so don't use @
            #lines[:,i_line] =  inputs@slopes[i_line,:].T + offsets[i_line]
            lines[:,i_line] = np.dot(inputs,slopes[i_line,:].T) + offsets[i_line]

        return lines


    def predict(self,inputs):

        if inputs.ndim==1:
            # Otherwise matrix multiplication below will not work
            inputs = np.atleast_2d(inputs).T
            
        lines = self.getLines(inputs)
            
        # Weight the values for each line with the normalized basis function activations  
        # Get the activations of the basis functions 
        activations = self.getActivations(inputs)
        
        outputs = (lines*activations).sum(axis=1)
        return outputs
        
    def isTrained(self):
        return len(self._model_params['offsets'])>0