import os
import sys
import numpy as np

# Add relative path, in case PYTHONPATH is not set
lib_path = os.path.abspath('../../../python/')
sys.path.append(lib_path)

from bbo.cost_function import CostFunction
from bbo.distribution_gaussian import DistributionGaussian
from bbo.updater import UpdaterCovarDecay
from bbo.run_optimization import runOptimization

class DemoCostFunctionDistanceToPoint(CostFunction):
    """ CostFunction in which the distance to a pre-defined point must be minimized."""

    def __init__(self,point,regularization_weight=1.0):
        """ Constructor.
        \param[in] point Point to which distance must be minimized.
        """
        self.point = np.asarray(point)
        self.regularization_weight = regularization_weight
        
  
    def evaluate(self,sample):
        # Compute distance from sample to point
        dist = np.linalg.norm(sample-self.point)
        # Regularization term
        if self.regularization_weight>0:
            regularization = self.regularization_weight*np.linalg.norm(sample)
            return [dist+regularization, dist, regularization]
        else:
            return [dist]

if __name__=="__main__":

    directory = None
    if (len(sys.argv)>1):
        directory = sys.argv[1]

    n_dims = 2
    minimum = np.full(n_dims,2.0)
    regul_weight = 1.0
    cost_function = DemoCostFunctionDistanceToPoint(minimum,regul_weight)

    
    mean_init  =  np.full(n_dims,5.0)
    covar_init =  4.0*np.eye(n_dims)
    distribution = DistributionGaussian(mean_init, covar_init)
    
    eliteness = 10
    weighting_method = 'PI-BB'
    covar_decay_factor = 0.8
    updater = UpdaterCovarDecay(eliteness,weighting_method,covar_decay_factor)
  
    n_samples_per_update = 20
    n_updates = 40

    import matplotlib.pyplot as plt
    fig = plt.figure(1,figsize=(15, 5))
    
    learning_curve = runOptimization(cost_function, distribution, updater, n_updates, n_samples_per_update,fig,directory)
    
    plt.show()


