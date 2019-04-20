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
import math
import os

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from bbo.DistributionGaussian import DistributionGaussian

class Updater:
    """ Virtual class for updating a Gaussian distribution with reward-weighted averaging.
    """
    def updateDistribution(distribution, samples, costs):
        """ Update a distribution with reward-weighted averaging.
        \param[in] distribution Distribution before the update
        \param[in] samples Samples in parameter space.
        \param[in] costs The cost of each sample.
        \return The updated distribution.
        """
        raise NotImplementedError('subclasses must override updateDistribution()!')


class UpdaterMean(Updater):
    """ Updater that updates the mean of the distribution only."""
            
    def __init__(self,eliteness = 10, weighting_method = 'PI-BB'):
        """ Initialize an UpdaterMean object.
        \param[in] eliteness The eliteness parameter (see costsToWeights(...))
        \param[in] weighting_method The weighting method ('PI-BB','CMA-ES','CEM', see costsToWeights(...))
        """
        self.eliteness = eliteness
        self.weighting_method = weighting_method

    def updateDistribution(self,distribution, samples, costs):
        """ Update a distribution with reward-weighted averaging.
        \param[in] distribution Distribution before the update
        \param[in] samples Samples in parameter space.
        \param[in] costs The cost of each sample.
        \return The updated distribution.
        """
    
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
    """ Updater that updates the mean of the distribution, and decays the covariance matrix of the distribution."""
            
    def __init__(self,eliteness = 10, weighting_method = 'PI-BB', covar_decay_factor = 0.8):
        """ Initialize an UpdaterCovarDecay object.
        \param[in] eliteness The eliteness parameter (see costsToWeights(...))
        \param[in] weighting_method The weighting method ('PI-BB','CMA-ES','CEM', see costsToWeights(...))
        \param[in] covar_decay_factor Factor with which to decay the covariance matrix (i.e. covar_decay_factor*covar_decay_factor*C at each update)
        """
        self.eliteness = eliteness
        self.weighting_method = weighting_method
        self.covar_decay_factor = covar_decay_factor

    def updateDistribution(self,distribution, samples, costs):
        """ Update a distribution with reward-weighted averaging.
        \param[in] distribution Distribution before the update
        \param[in] samples Samples in parameter space.
        \param[in] costs The cost of each sample.
        \return The updated distribution.
        """
    
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
    """ Updater that updates the mean of the distribution, and uses covariance matrix adaptation to update the covariance matrix of the distribution."""
            
    def __init__(self,eliteness = 10, weighting_method = 'PI-BB',diagonal_max=None,diagonal_min=None,diag_only=False,learning_rate=1.0):
        """ Constructor
        \param[in] eliteness The eliteness parameter ('mu' in CMA-ES, 'h' in PI^2)
        \param[in] weighting_method ('PI-BB' = PI^2 style weighting)
        \param[in] base_level Small covariance matrix that is added after each update to avoid premature convergence
        \param[in] diag_only Update only the diagonal of the covariance matrix (true) or the full matrix (false)
        \param[in] learning_rate Low pass filter on the covariance updates. In range [0.0-1.0] with 0.0 = no updating, 1.0  = complete update by ignoring previous covar matrix. 
        """
        self.eliteness = eliteness
        self.weighting_method = weighting_method
        self.diagonal_max = diagonal_max
        self.diagonal_min = diagonal_min
        self.diag_only = diag_only
        if (learning_rate>1.0):
            learning_rate=1.0
        elif (learning_rate<0.0):
            learning_rate=0.0
        self.learning_rate = learning_rate

    def updateDistribution(self,distribution, samples, costs):
        """ Update a distribution with reward-weighted averaging.
        \param[in] distribution Distribution before the update
        \param[in] samples Samples in parameter space.
        \param[in] costs The cost of each sample.
        \return The updated distribution.
        """
        
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
            
        # Set a maximum value for the diagonal to avoid too much exploration
        if (self.diagonal_max is not None):
            is_scalar_max = np.isscalar(self.diagonal_max)
            if is_scalar_max:
                level_max = pow(self.diagonal_max,2)
            for ii in range(n_dims):
                if not is_scalar_max:
                    level_max = pow(self.diagonal_max[ii],2)
                if covar_new[ii,ii]>level_max:
                    covar_new[ii,ii]=level_max
                    
        # Set a minimum value for the diagonal to avoid pre-mature convergence
        if (self.diagonal_min is not None):
            is_scalar_min = np.isscalar(self.diagonal_min)
            if is_scalar_min:
                level_min = pow(self.diagonal_min,2)
            for ii in range(n_dims):
                if not is_scalar_min:
                    level_min = pow(self.diagonal_min[ii],2)
                if covar_new[ii,ii]<level_min:
                    covar_new[ii,ii]=level_min
                    
                    
        # Update the covariance matrix
        distribution_new = DistributionGaussian(mean_new, covar_new)
    
        return distribution_new, weights

def costsToWeights(costs, weighting_method, eliteness):
    """ Convert costs to weights using different weighting methods.
    \param[in] costs A vector of costs.
    \param[in] weighting_method The weighting method ('PI-BB','CMA-ES','CEM')
    \param[in] eliteness The eliteness parameter (h in PI-BB, mu in CMA-ES)
    \return A vector of weights (they always sum to 1).
    """
    
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


if __name__=="__main__":
    eliteness = 10
    weighting_method = 'PI-BB'
    covar_decay_factor = 0.8
    updater = UpdaterCovarDecay(eliteness,weighting_method,covar_decay_factor)
    
    diagonal_min = 0.1
    diagonal_max = 1.0
    diag_only=False
    learning_rate=1.0
    updater = UpdaterCovarAdaptation(eliteness, weighting_method,diagonal_max,diagonal_min,diag_only,learning_rate)

    mu  = np.array([2,4])
    cov = np.array([[0.3,0.0],[0.0,0.5]])
    distribution = DistributionGaussian(mu,cov)
    
    n_samples = 10
    samples = distribution.generateSamples(n_samples)
    costs = abs(samples[:,0]) + abs(samples[:,1]) # Manhattan distance
    
    (new_distribution, w) = updater.updateDistribution(distribution, samples, costs)
    print(distribution.covar)
    print(new_distribution.covar)
