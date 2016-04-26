import numpy as np
import math
import os

from update_summary import UpdateSummary

def runOptimization(cost_function, initial_distribution, updater, n_updates, n_samples_per_update):

    
    distribution = initial_distribution
  
    # Optimization loop
    costs = np.full(n_samples_per_update,0.0)
    update_summaries = []
    for i_update in range(n_updates): 
        
        # 0. Get cost of current distribution mean
        cost_eval = cost_function.evaluate(distribution.mean)
        
        # 1. Sample from distribution
        samples = distribution.generateSamples(n_samples_per_update)
    
        # 2. Evaluate the samples
        for i_sample in range(n_samples_per_update):
            costs[i_sample] = cost_function.evaluate(samples[i_sample,:])
      
        # 3. Update parameters
        distribution_new, weights = updater.updateDistribution(distribution, samples, costs)
        
        # Bookkeeping
        update_summary = UpdateSummary(distribution,samples,cost_eval,costs,weights,distribution_new)
        update_summaries.append(update_summary)
        
        # Distribution is new distribution
        distribution = distribution_new
        
    return update_summaries
