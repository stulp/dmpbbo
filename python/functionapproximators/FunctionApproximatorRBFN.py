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

from functionapproximators.BasisFunction import Gaussian
from functionapproximators.leastSquares import *

class FunctionApproximatorRBFN:
    
    def __init__(self,n_basis_functions_per_dim, intersection_height=0.5, regularization=0.0):
        
        self.meta_n_basis_functions_per_dim_ = n_basis_functions_per_dim
        self.meta_intersection_height_ = intersection_height
        self.meta_regularization_ = regularization

        self.model_centers_ = None
        self.model_widths_ = None
        self.model_weights_ = None
        self.is_trained_ = False
        
    def train(self,inputs,targets):

        
        # Determine the centers and widths of the basis functions, given the range of the input data
        min_vals = np.asscalar(inputs.min(axis=0))
        max_vals = np.asscalar(inputs.max(axis=0))

        n_centers = self.meta_n_basis_functions_per_dim_
        centers = np.linspace(min_vals,max_vals,n_centers)
        widths = np.ones(n_centers)
        if n_centers>1:
            # Consider two neighbouring basis functions, exp(-0.5(x-c0)^2/w^2) and exp(-0.5(x-c1)^2/w^2)
            # Assuming the widths are the same for both, they are certain to intersect at x = 0.5(c0+c1)
            # And we want the activation at x to be 'intersection'. So
            #            y = exp(-0.5(x-c0)^2/w^2)
            # intersection = exp(-0.5((0.5(c0+c1))-c0)^2/w^2)
            # intersection = exp(-0.5((0.5*c1-0.5*c0)^2/w^2))
            # intersection = exp(-0.5((0.5*(c1-c0))^2/w^2))
            # intersection = exp(-0.5(0.25*(c1-c0)^2/w^2))
            # intersection = exp(-0.125((c1-c0)^2/w^2))
            #            w = sqrt((c1-c0)^2/-8*ln(intersection))
            for cc in range(n_centers-1):
                w = np.sqrt(np.square(centers[cc+1]-centers[cc])/(-8*np.log(self.meta_intersection_height_)))
                widths[cc] = w
                
            widths[n_centers-1] = widths[n_centers-2]
       
        # Get the activations of the basis functions 
        self.model_widths_ = widths
        self.model_centers_ = centers
        activations = self.getActivations(inputs)

        
        # Perform one weighted least squares regression for each kernel
        use_offset = False
        reg = self.meta_regularization_
        self.model_weights_ = leastSquares(activations,targets,use_offset,reg)
    
        self.is_trained_ = True
        

    def getActivations(self,inputs):
        normalize_activations = False
        activations = Gaussian.activations(self.model_centers_,self.model_widths_,inputs,normalize_activations)
        return activations

    def predict(self,inputs):
        #print('====================\npredict')
        #print(inputs.shape)

        if inputs.ndim==1:
            # Otherwise matrix multiplication below will not work
            inputs = np.atleast_2d(inputs).T
            
        # Get the activations of the basis functions 
        activations = self.getActivations(inputs)
        n_basis_functions = activations.shape[1]
        
        outputs = activations
        for ii in range(n_basis_functions):
            outputs[:,ii] = outputs[:,ii]*self.model_weights_[ii]
            
        outputs = outputs.sum(axis=1)
            
        return outputs
        
    def isTrained(self):
        return self.is_trained_

