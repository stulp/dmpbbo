# This file is part of DmpBbo, a set of libraries and programs for the 
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2018, 2022 Freek Stulp
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
        """Initialiaze an RBNF function approximator.

        Args:
            n_basis_functions_per_dim Number of basis functions per input dimension.
            intersection_height The relative value at which two neighbouring basis functions will intersect (default=0.7)
            regularization Regularization parameter (default=0.0)
            """
        
        n_bfs_per_dim = np.atleast_1d(n_basis_functions_per_dim)
        
        meta_params = {
            'n_basis_functions_per_dim': n_bfs_per_dim,
            'intersection_height': intersection_height,
            'regularization': regularization,
        }

        super().__init__(meta_params)

    def dim_input(self):
        return self._model_params['centers'].shape[1]
        
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
    

    def isTrained(self):
        """Determine whether the function approximator has already been trained with data or not.
        
        Returns:
            bool: True if the function approximator has already been trained, False otherwise.
        """
        if not self._model_params:
            return False
        if not 'weights' in self._model_params:
            return False
        return True
        
    def getActivations(self,inputs):
        """Get the activations of the basis functions.
        
        Uses the centers and widths in the model parameters.
        
        Args:
            inputs (numpy.ndarray): Input values of the query.
        """
        normalize = False
        centers = self._model_params['centers']
        widths = self._model_params['widths']
        activations = Gaussian.activations(centers,widths,inputs,normalize)
        return activations


    def predict(self,inputs):
        """Implements abstract function FunctionApproximator
        """
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