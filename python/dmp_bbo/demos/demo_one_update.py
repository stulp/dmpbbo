import os
import sys
import numpy as np

lib_path = os.path.abspath('../../bbo')
sys.path.append(lib_path)
lib_path = os.path.abspath('../')
sys.path.append(lib_path)

from distribution_gaussian import DistributionGaussian
from task import Task
from updater import UpdaterCovarDecay
#from update_summary import UpdateSummary, extractLearningCurve

from run_one_update import runOptimizationTaskOneUpdate

def targetFunction(a, c, inputs):
    """ Target function \f$ y = a*x^2 + c \f$
    \param[in] a a in \f$ y = a*x^2 + c \f$
    \param[in] c c in \f$ y = a*x^2 + c \f$
    \param[in] inputs x in \f$ y = a*x^2 + c \f$
    \return outputs y in \f$ y = a*x^2 + c \f$
    """
    
    # Compute a*x^2 + c
    outputs = [a*x*x + c for x in inputs]
    return np.array(outputs)

class DemoTaskApproximateQuadraticFunction(Task):
    """
    The task is to choose the parameters a and c such that the function \f$ y = lib_path = os.path.abspath('../')
sys.path.append(lib_path)
a*x^2 + c \f$ best matches a set of target values y_target for a set of input values x
    """

    def __init__(self, a, c, inputs):
        """ Constructor
        \param[in] a a in \f$ y = a*x^2 + c \f$
        \param[in] c c in \f$ y = a*x^2 + c \f$
        \param[in] inputs x in \f$ y = a*x^2 + c \f$
        """
        self.inputs = inputs;
        self.targets = targetFunction(a,c,inputs);
  
    def evaluateRollout(self, cost_vars):
        """ Cost function
        \param[in] cost_vars y in \f$ y = a*x^2 + c \f$
        \return costs Costs of the cost_vars
        """
        diff_square = np.square(cost_vars-self.targets)
        costs = [np.mean(diff_square)]
        return costs



def doOneUpdate(directory):
    inputs = np.linspace(-1.5,1.5,21);
    a = 2.0
    c = -1.0
    n_params = 2

    task = DemoTaskApproximateQuadraticFunction(a,c,inputs)
  
    mean_init  =  np.full(n_params,0.5)
    covar_init =  0.25*np.eye(n_params)
    initial_distribution = DistributionGaussian(mean_init, covar_init)
  
    eliteness = 10
    weighting_method = 'PI-BB'
    covar_decay_factor = 0.8
    updater = UpdaterCovarDecay(eliteness,weighting_method,covar_decay_factor)
  
    n_samples_per_update = 10
    runOptimizationTaskOneUpdate(task, initial_distribution, updater, n_samples_per_update,directory)


if __name__=="__main__":
    # See if input directory was passed
    if (len(sys.argv)==2):
        directory = str(sys.argv[1])
    else:
        print '\nUsage: '+sys.argv[0]+' <directory>\n';
        sys.exit()
  
    doOneUpdate(directory)
    


