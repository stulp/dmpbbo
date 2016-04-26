import numpy as np
from distribution_gaussian import DistributionGaussian

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
    
    if weighting_method == 'PI-BB':
        # PI^2 style weighting: continuous, cost exponention
        h = eliteness # In PI^2, eliteness parameter is known as "h"
        costs_range = max(costs)-min(costs)
        if costs_range==0:
            weights = np.full(size(costs),1.0)
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
