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

class FunctionApproximatorRBFN(FunctionApproximator):
    
    def __init__(self,n_basis_functions_per_dim, intersection_height=0.7, regularization=0.0):
        
        meta_params = {
            'n_basis_functions_per_dim': n_basis_functions_per_dim,
            'intersection_height': intersection_height,
            'regularization': regularization,
        }

        super().__init__(meta_params)

    def getSelectableParameters(self):
        return ['centers','widths','weights']

    def getSelectableParametersRecommended(self):
        return ['weights']
        
    def train(self,inputs,targets):
        
        # Determine the centers and widths of the basis functions, given the input data range
        min_vals = inputs.min(axis=0)
        max_vals = inputs.max(axis=0)
        n_bfs_per_dim = self._meta_params['n_basis_functions_per_dim']
        height = self._meta_params['intersection_height']
        (centers,widths) = getCentersAndWidths(min_vals, max_vals, n_bfs_per_dim, height)

        # Get the activations of the basis functions 
        self._model_params = {}
        self._model_params['centers'] = centers
        self._model_params['widths'] = widths
        
        # Perform one least squares regression
        use_offset = False
        reg = self._meta_params['regularization']
        activations = self.getActivations(inputs)
        weights = leastSquares(activations,targets,use_offset,reg)
        self._model_params['weights'] = np.atleast_2d(weights).T
    

    def getActivations(self,inputs):
        normalize_activations = False
        centers = self._model_params['centers']
        widths = self._model_params['widths']
        activations = Gaussian.activations(centers,widths,inputs,normalize_activations)
        return activations

    def predict(self,inputs):
        if not self.isTrained():
            raise ValueError('FunctionApproximator is not trained.')

        if inputs.ndim==1:
            # Otherwise matrix multiplication below will not work
            inputs = np.atleast_2d(inputs).T
            
        # Get the activations of the basis functions 
        activations = self.getActivations(inputs)
        n_basis_functions = activations.shape[1]
        
        outputs = activations
        for ii in range(n_basis_functions):
            outputs[:,ii] = outputs[:,ii]*self._model_params['weights'][ii]
            
        return outputs.sum(axis=1)