import os
import sys
import numpy as np

lib_path = os.path.abspath('../')
sys.path.append(lib_path)

from distribution_gaussian import DistributionGaussian
from updater import UpdaterCovarDecay
from run_optimization import runOptimization
from update_summary import UpdateSummary, extractLearningCurve
from cost_function import CostFunction


class DemoCostFunctionDistanceToPoint(CostFunction):
    """ CostFunction in which the distance to a pre-defined point must be minimized."""

    def __init__(self,point):
        """ Constructor.
        \param[in] point Point to which distance must be minimized.
        """
        self.point = point
  
    def evaluate(self,samples):
        n_dims = len(self.point) # Dimensionality of point
        
        # Very un-Pythonic. Sorry, written on train without references
        if samples.ndim==1: # Only one sample
            assert len(samples)==n_dims
            costs = np.linalg.norm(samples-self.point)
        else:
            assert samples.shape[1]==n_dims
            costs = [ np.linalg.norm(sample-self.point) for sample in samples]
        
        return costs

def testRunEvolutionaryOptimization():

    n_dims = 2
    minimum = np.full(n_dims,0.0)
    cost_function = DemoCostFunctionDistanceToPoint(minimum)

    
    mean_init  =  np.full(n_dims,5.0)
    covar_init =  4.0*np.eye(n_dims)
    distribution = DistributionGaussian(mean_init, covar_init)
    
    eliteness = 10
    weighting_method = 'PI-BB'
    covar_decay_factor = 0.8
    updater = UpdaterCovarDecay(eliteness,weighting_method,covar_decay_factor)
  
    n_samples_per_update = 20
    n_updates = 40
    update_summaries = runOptimization(cost_function, distribution, updater, n_updates, n_samples_per_update)


    import matplotlib.pyplot as plt
    import bbo_plotting
    fig = plt.figure(1,figsize=(15, 5))
    axs = [ fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]
    bbo_plotting.plotEvolutionaryOptimization(update_summaries,axs)
    plt.show()

if __name__=="__main__":
  
  testRunEvolutionaryOptimization()

