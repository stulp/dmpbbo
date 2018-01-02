# This file is part of DmpBbo, a set of libraries and programs for the 
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
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


import numpy as np
import sys
import os

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from bbo.distribution_gaussian import DistributionGaussian

class Updater:
    def updateDistribution(distribution, samples, costs):
        raise NotImplementedError('subclasses must override updateDistribution()!')


class UpdaterMean(Updater):
            
    def __init__(self,eliteness = 10, weighting_method = 'PI-BB'):
        self.eliteness = eliteness
        self.weighting_method = weighting_method

    def updateDistribution(self,distribution, samples, costs):
    
        weights = costsToWeights(costs,self.weighting_method,self.eliteness);
    
        # Compute new mean with reward-weighed averaging
        # mean    = 1 x n_dims
        # weights = 1 x n_samples
        # samples = n_samples x n_dims
        mean_new = np.average(samples,0,weights)
      
        # Update the covariance matrix
        distribution_new = DistributionGaussian(mean_new, distribution.covar)
    
        return distribution_new, weights
        
class UpdaterCovarDecay(Updater):
            
    def __init__(self,eliteness = 10, weighting_method = 'PI-BB', covar_decay_factor = 0.8):
        self.eliteness = eliteness
        self.weighting_method = weighting_method
        self.covar_decay_factor = covar_decay_factor

    def updateDistribution(self,distribution, samples, costs):
    
        weights = costsToWeights(costs,self.weighting_method,self.eliteness);
    
        # Compute new mean with reward-weighed averaging
        # mean    = 1 x n_dims
        # weights = 1 x n_samples
        # samples = n_samples x n_dims
        mean_new = np.average(samples,0,weights)
        decay = self.covar_decay_factor
        covar_new = decay*decay*distribution.covar
      
        # Update the covariance matrix
        distribution_new = DistributionGaussian(mean_new, covar_new)
    
        return distribution_new, weights

class UpdaterCovarAdaptation(Updater):
            
    def __init__(self,eliteness = 10, weighting_method = 'PI-BB',base_level_diagonal=None,diag_only=False,learning_rate=1.0):
        self.eliteness = eliteness
        self.weighting_method = weighting_method
        self.base_level_diagonal = base_level_diagonal
        self.diag_only = diag_only
        if (learning_rate>1.0):
            learning_rate=1.0
        elif (learning_rate<0.0):
            learning_rate=0.0
        self.learning_rate = learning_rate

    def updateDistribution(self,distribution, samples, costs):
        
        mean_cur = distribution.mean
        covar_cur = distribution.covar
        n_samples = samples.shape[0]
        n_dims = samples.shape[1]
    
        weights = costsToWeights(costs,self.weighting_method,self.eliteness);
    
        # Compute new mean with reward-weighed averaging
        # mean    = 1 x n_dims
        # weights = 1 x n_samples
        # samples = n_samples x n_dims
        mean_new = np.average(samples,0,weights)
        
        #np.set_printoptions(precision=4, suppress=True)
        eps = samples - np.tile(mean_cur,(n_samples,1))
        weights_tiled = np.tile(np.asarray([weights]).transpose(),(1,n_dims))
        weighted_eps = np.multiply(weights_tiled,eps)
        covar_new = np.dot(weighted_eps.transpose(),eps)
    
        # Remove non-diagonal values
        if (self.diag_only):
            diag_vec = np.diag(covar_new)
            covar_new = np.diag(diag_vec)
        
        # Low-pass filter for covariance matrix, i.e. weight between current
        # and new covariance matrix.
        
        if (self.learning_rate<1.0):
            lr = self.learning_rate # For legibility
            covar_new = (1-lr)*covar_cur + lr*covar_new;
            
        # Add a base_level to avoid pre-mature convergence
        if (self.base_level_diagonal is not None):
            for ii in range(n_dims):
                if covar_new[ii,ii]<self.base_level_diagonal[ii]:
                    covar_new[ii,ii]=self.base_level_diagonal[ii]

        # Update the covariance matrix
        distribution_new = DistributionGaussian(mean_new, covar_new)
    
        return distribution_new, weights

def costsToWeights(costs, weighting_method, eliteness):
    
    # Costs can be a 2D array or a list of lists. In this case, the first
    # column is the sum of the other columns (which contain the different cost
    # components). In this  case, we should use only the first column.
    costs = np.asarray([np.atleast_1d(x)[0] for x in costs])

    #np.set_printoptions(precision=4, suppress=True)
    if weighting_method == 'PI-BB':
        # PI^2 style weighting: continuous, cost exponention
        h = eliteness # In PI^2, eliteness parameter is known as "h"
        costs_range = max(costs)-min(costs)
        if costs_range==0:
            weights = np.full(costs.shape,1.0)
        else:
            costs_norm = np.asarray([-h*(x-min(costs))/costs_range for x in costs])
            weights = np.exp(costs_norm)

    elif weighting_method=='CEM' or weighting_method=='CMA-ES':
        # CEM/CMA-ES style weights: rank-based, uses defaults
        mu = eliteness # In CMA-ES, eliteness parameter is known as "mu"
        indices = np.argsort(costs)
        weights = np.full(costs.size,0.0)
        if weighting_method=='CEM':
            # CEM
            weights[indices[0:mu]] = 1.0/mu
        else:
            # CMA-ES
            for ii in range(mu):
                weights[indices[ii]] = np.log(mu+0.5)-np.log(ii+1)
            
    else:
        print("WARNING: Unknown weighting method '", weighting_method, "'. Calling with PI-BB weighting."); 
        return costsToWeights(costs, 'PI-BB', eliteness);
  
    #// Relative standard deviation of total costs
    #double mean = weights.mean();
    #double std = sqrt((weights.array()-mean).pow(2).mean());
    #double rel_std = std/mean;
    #if (rel_std<1e-10)
    #{
    #    // Special case: all costs are the same
    #    // Set same weights for all.
    #    weights.fill(1);
    #}

    # Normalize weights
    weights = weights/sum(weights)
    
    return weights
