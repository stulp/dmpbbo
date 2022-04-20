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
        
        if isinstance(n_basis_functions_per_dim,int):
            n_basis_functions_per_dim = [n_basis_functions_per_dim]
        
        meta_params = {
            'n_basis_functions_per_dim': n_basis_functions_per_dim,
            'intersection_height': intersection_height,
            'regularization': regularization,
        }

        super().__init__(meta_params)

    def dim_input(self):
        if not isTrained():
            raise ValueError("Can only call dim_input on trained function approximator.")
        return self._model_params['centers'].shape[1]
        
    def getSelectableParameters(self):
        return ['centers','widths','slopes','offsets']

    def getSelectableParametersRecommended(self):
        return ['slopes','offsets']
        
    def train(self,inputs,targets):

        # Determine the centers and widths of the basis functions, given the input data range
        min_vals = inputs.min(axis=0)
        max_vals = inputs.max(axis=0)
        n_bfs_per_dim = self._meta_params['n_basis_functions_per_dim']
        height = self._meta_params['intersection_height']
        (centers,widths) = getCentersAndWidths(min_vals, max_vals, n_bfs_per_dim, height)
       
        # Get the activations of the basis functions 
        self._model_params = {}
        self._model_params['widths'] = widths
        self._model_params['centers'] = centers

        # Parameters for the weighted least squares regressions
        use_offset = True
        n_kernels = np.prod(n_bfs_per_dim)
        n_dims = centers.shape[1]
        n_betas = n_dims
        if (use_offset):
            n_betas += 1
        betas = np.ones([n_kernels,n_betas])
        
        # Perform one weighted least squares regression for each kernel
        activations = self.getActivations(inputs)
        reg = self._meta_params['regularization']
        for i_kernel in range(n_kernels):
            weights = activations[:,i_kernel]
            beta = weightedLeastSquares(inputs,targets,weights,use_offset,reg)
            betas[i_kernel,:] = beta.T
    
        self._model_params['offsets'] = np.atleast_2d(betas[:,-1]).T
        self._model_params['slopes'] = np.atleast_2d(betas[:,0:-1])

    def isTrained(self):
        """Determine whether the function approximator has already been trained with data or not.
        
        Returns:
            bool: True if the function approximator has already been trained, False otherwise.
        """
        if not self._model_params:
            return False
        if not 'offsets' in self._model_params:
            return False
        return True
        
    def getActivations(self,inputs):
        """Get the activations of the basis functions.
        
        Uses the centers and widths in the model parameters.
        
        Args:
            inputs (numpy.ndarray): Input values of the query.
        """
        normalize = True
        centers = self._model_params['centers']
        widths = self._model_params['widths']
        activations = Gaussian.activations(centers,widths,inputs,normalize)
        return activations
        
    def getLines(self,inputs):
        if inputs.ndim==1:
            # Otherwise matrix multiplication below will not work
            inputs = np.atleast_2d(inputs).T
        
        slopes = self._model_params['slopes']
        offsets = self._model_params['offsets']
        
        # Compute the line segments
        n_lines = offsets.size 
        n_samples = inputs.shape[0]
        lines = np.zeros([n_samples,n_lines])
        for i_line in range(n_lines):
            # Apparently, not everybody has python3.5 installed, so don't use @
            #lines[:,i_line] =  inputs@slopes[i_line,:].T + offsets[i_line]
            lines[:,i_line] = np.dot(inputs,slopes[i_line,:].T) + offsets[i_line]

        return lines


    def predict(self,inputs):
        if not self.isTrained():
            raise ValueError('FunctionApproximator is not trained.')

        if inputs.ndim==1:
            # Otherwise matrix multiplication below will not work
            inputs = np.atleast_2d(inputs).T
            
        lines = self.getLines(inputs)
            
        # Weight the values for each line with the normalized basis function activations  
        # Get the activations of the basis functions 
        activations = self.getActivations(inputs)
        
        outputs = (lines*activations).sum(axis=1)
        return outputs