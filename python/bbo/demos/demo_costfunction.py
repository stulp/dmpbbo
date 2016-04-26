import os
import sys
import numpy as np

lib_path = os.path.abspath('../../../python/')
sys.path.append(lib_path)


from bbo.distribution_gaussian import DistributionGaussian
from bbo.updater import UpdaterCovarDecay
from bbo.run_optimization import runOptimization
from bbo.cost_function import CostFunction

from bbo.bbo_plotting import plotEvolutionaryOptimization


class DemoCostFunctionDistanceToPoint(CostFunction):
    """ CostFunction in which the distance to a pre-defined point must be minimized."""

    def __init__(self,point):
        """ Constructor.
        \param[in] point Point to which distance must be minimized.
        """
        self.point = np.asarray(point)
  
    def evaluate(self,sample):
        # Compute distance from sample to point
        return np.linalg.norm(sample-self.point) 

if __name__=="__main__":

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
    fig = plt.figure(1,figsize=(15, 5))
    axs = [ fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]
    plotEvolutionaryOptimization(update_summaries,axs)
    plt.show()


