import os
import sys
import numpy as np
import matplotlib.pyplot as plt

lib_path = os.path.abspath('../../../python')
sys.path.append(lib_path)

from bbo.distribution_gaussian import DistributionGaussian
from bbo.updater import UpdaterCovarDecay
from dmp_bbo.task import Task

from dmp_bbo.run_one_update import runOptimizationTaskOneUpdate
from dmp_bbo.dmp_bbo_plotting import plotOptimizationRollouts

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
        # Compute a*x^2 + c
        self.targets = [a*x*x + c for x in inputs]
  
    def evaluateRollout(self, cost_vars):
        """ Cost function
        \param[in] cost_vars y in \f$ y = a*x^2 + c \f$
        \return costs Costs of the cost_vars
        """
        diff_square = np.square(cost_vars-self.targets)
        costs = [np.mean(diff_square)]
        return costs

    def plotRollout(self,cost_vars,ax):
        line_handles = ax.plot(self.inputs,cost_vars.T,linewidth=0.5)
        ax.plot(self.inputs,self.targets,'-o',color='k',linewidth=2)
        return line_handles


if __name__=="__main__":
    # See if input directory was passed
    if (len(sys.argv)>=2):
        directory = str(sys.argv[1])
    else:
        print '\nUsage: '+sys.argv[0]+' <directory> [plot_results]\n';
        sys.exit()

    plot_results = False
    if (len(sys.argv)>=3):
        plot_results = True
        
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
    
    i_update = runOptimizationTaskOneUpdate(directory, task, initial_distribution, updater, n_samples_per_update)

    i_update -= 1
    if plot_results and i_update>1:
        # Plot the optimization results (from the files saved to disk)
        fig = plt.figure(1,figsize=(15, 5))
        plotOptimizationRollouts(directory,fig,task.plotRollout)
        plt.show()

