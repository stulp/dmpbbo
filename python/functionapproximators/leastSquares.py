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

import numpy as np

def leastSquares(inputs, targets, use_offset=True, regularization=0.0):
    weights = np.ones(inputs.shape[0])
    return weightedLeastSquares(inputs,targets,weights,use_offset,regularization)


def weightedLeastSquares(inputs, targets, weights, use_offset=True, regularization=0.0):
    
    
    # Inputs and targets should have "n_samples" rows. If not, transpose them.
    #if inputs.shape[0] != n_samples and inputs.shape[1] == n_samples:
    #    inputs = inputs.T
    #if targets.shape[0] != n_samples and targets.shape[1] == n_samples:
    #    targets = targets.T
    
    n_samples = weights.size

    # Make the design matrix
    if use_offset:
        # Add a column with 1s
        X = np.column_stack((inputs,np.ones(n_samples)))
        
    else:
        X = inputs
  
  
    # Weights matrix
    W = np.diagflat(weights)
    
    
    # Regularization matrix
    n_input_dims = 1
    if X.ndim>1:
        n_input_dims = X.shape[1]
    Gamma = regularization*np.identity(n_input_dims) 

    if X.ndim==1:
        # Otherwise matrix multiplication below will not work
        X = np.atleast_2d(X).T

    # Least squares then is a one-liner
    # Apparently, not everybody has python3.5 installed, so don't use @
    # betas = np.linalg.inv(X.T@W@X + Gamma)@X.T@W@targets
    # In python<=3.4, it is not a one-liner
    to_invert = np.dot(np.dot(X.T,W),X) + Gamma
    betas = np.dot(np.dot(np.dot(np.linalg.inv(to_invert),X.T),W),targets)
    
    return betas


def linearPrediction(inputs,betas):

    if inputs.ndim==1:
        # Otherwise matrix multiplication below will not work
        inputs = np.atleast_2d(inputs).T
    n_input_dims = inputs.shape[1]
        
    n_beta = betas.size
    
    if n_input_dims==n_beta:
        # Apparently, not everybody has python3.5 installed, so don't use @
        #outputs = inputs@betas
        outputs = np.dot(inputs,betas)
    else:
        # There is an offset (AKA bias or intercept)
        if  n_input_dims+1 != n_beta:
            raise ValueError(f'betas is of the wrong size (is {n_beta}, should be {n_inputs_dims+1})')
        # Apparently, not everybody has python3.5 installed, so don't use @
        #outputs = inputs@betas[0:n_beta-1] + betas[-1]
        outputs = np.dot(inputs,betas[0:n_beta-1]) + betas[-1]
    
    return outputs
