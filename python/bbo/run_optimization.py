import numpy as np
import math
import os
from pylab import sqrt

from bbo_plotting import plotUpdate, plotCurve

def runOptimization(cost_function, initial_distribution, updater, n_updates, n_samples_per_update,fig=None,directory=None):
    
    distribution = initial_distribution

    learning_curve = np.zeros((n_updates, 3))

    if fig:
        ax = fig.add_subplot(131)

    
    # Optimization loop
    for i_update in range(n_updates): 
        
        # 0. Get cost of current distribution mean
        cost_eval = cost_function.evaluate(distribution.mean)
        
        # 1. Sample from distribution
        samples = distribution.generateSamples(n_samples_per_update)
    
        # 2. Evaluate the samples
        costs = []
        for i_sample in range(n_samples_per_update):
            costs.append(cost_function.evaluate(samples[i_sample,:]))
      
        # 3. Update parameters
        distribution_new, weights = updater.updateDistribution(distribution, samples, costs)
        
        
        # Bookkeeping and plotting
        
        # Update learning curve
        # How many samples so far?  
        learning_curve[i_update,0] = i_update*n_samples_per_update
        # Cost of evaluation
        learning_curve[i_update,1] = cost_eval
        # Exploration magnitude
        learning_curve[i_update,2] = sqrt(distribution.maxEigenValue()); 

        # Plot summary of this update
        if fig:
            highlight = (i_update==0)
            plotUpdate(distribution,cost_eval,samples,costs,weights,distribution_new,ax,highlight)
        
        # Distribution is new distribution
        distribution = distribution_new;
        
        
    # Plot learning curve
    if fig:
        axs = [ fig.add_subplot(132), fig.add_subplot(133)]
        plotCurve(learning_curve,axs)

    # Save learning curve to file, if necessary
    if directory:
        np.savetxt(directory+'/learning_curve.txt',learning_curve)
        
        
    return learning_curve
