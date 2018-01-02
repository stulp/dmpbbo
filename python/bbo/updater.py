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

def costsToWeights(costs, weighting_method, eliteness):
    
    # Costs can be a 2D array or a list of lists. In this case, the first
    # column is the sum of the other columns (which contain the different cost
    # components). In this  case, we should use only the first column.
    costs = [np.atleast_1d(x)[0] for x in costs]

    if weighting_method == 'PI-BB':
        # PI^2 style weighting: continuous, cost exponention
        h = eliteness # In PI^2, eliteness parameter is known as "h"
        costs_range = max(costs)-min(costs)
        if costs_range==0:
            weights = np.full(costs.shape,1.0)
        else:
            costs_norm = [-h*(x-min(costs))/costs_range for x in costs]
            weights = np.exp(costs_norm)
            
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
