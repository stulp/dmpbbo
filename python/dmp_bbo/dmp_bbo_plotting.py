import numpy as np
import os
import sys

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from rollout import Rollout, loadRolloutFromDirectory

from bbo.distribution_gaussian import DistributionGaussian
from bbo.bbo_plotting import plotUpdate, plotCurve, setColor, saveUpdate


def plotOptimizationRollouts(directory,fig,plotRollout=None,plot_all_rollouts=False):
    
    if not fig:    
        fig = plt.figure(1,figsize=(9, 4))

    # Determine number of updates
    n_updates = 1
    update_dir = '%s/update%05d' % (directory, n_updates)
    while os.path.exists(update_dir+"/distribution_new_mean.txt"):
        n_updates += 1
        update_dir = '%s/update%05d' % (directory, n_updates)
    n_updates -= 1
    
    if n_updates<2:
        return None
    
    learning_curve = np.zeros((n_updates, 3))
    
    
    for i_update in range(1,n_updates):
    
        update_dir = '%s/update%05d' % (directory, i_update)
    
        # Read data
        try:
            n_parallel = np.loadtxt(update_dir+"/n_parallel.txt")
        except IOError:
            n_parallel = 1
        
        if n_parallel==1:
            mean = np.loadtxt(update_dir+"/distribution_mean.txt")
            covar = np.loadtxt(update_dir+"/distribution_covar.txt")
            distributions = [DistributionGaussian(mean,covar)]
            
            try:
                mean = np.loadtxt(update_dir+"/distribution_new_mean.txt")
                covar = np.loadtxt(update_dir+"/distribution_new_covar.txt")
                distributions_new = [DistributionGaussian(mean,covar)]
            except IOError:
                distributions_new =[]
    
        else:
            distributions = []
            distributions_new = []
            for i_parallel in range(n_parallel):
                cur_file = '%s/distribution_%03d' % (update_dir, i_parallel)
                mean = np.loadtxt(cur_file+'_mean.txt')
                covar = np.loadtxt(cur_file+'_covar.txt')
                distributions.append(DistributionGaussian(mean,covar))
                
            try:
                for i_parallel in range(n_parallel):
                    cur_file = '%s/distribution_new_%03d' % (update_dir, i_parallel)
                    mean = np.loadtxt(cur_file+'_mean.txt')
                    covar = np.loadtxt(cur_file+'_covar.txt')
                    distributions_new.append(DistributionGaussian(mean,covar))
            except IOError:
                distributions_new =[]
    
        
        try:
            samples = np.loadtxt(update_dir+"/samples.txt")
        except IOError:
            samples = None
        costs = np.loadtxt(update_dir+"/costs.txt")
        weights = np.loadtxt(update_dir+"/weights.txt")
    
        
        try:
            cost_eval = np.loadtxt(update_dir+"/cost_eval.txt")
        except IOError:
            cost_eval = None
    
        n_rollouts = len(weights)
        rollouts = []
        for i_rollout in range(n_rollouts):
            rollout_dir = '%s/rollout%03d' % (update_dir, i_rollout+1)
            rollouts.append(loadRolloutFromDirectory(rollout_dir))
            
        rollout_eval = loadRolloutFromDirectory(update_dir+'/rollout_eval/')
    
        # Update learning curve 
        # How many samples so far?  
        learning_curve[i_update,0] = learning_curve[i_update-1,0] + n_rollouts
        # Cost of evaluation
        if cost_eval!=None:
            learning_curve[i_update,1] = np.atleast_1d(cost_eval)[0]
        # Exploration magnitude
        learning_curve[i_update,2] = 0.0
        for distr in distributions:
            learning_curve[i_update,2] += np.sqrt(distr.maxEigenValue()); 
        
        n_subplots = 3
        i_subplot = 1
        if plotRollout:
            n_subplots = 4
            ax_rollout = fig.add_subplot(1,n_subplots,i_subplot)
            i_subplot += 1
            h = plotRollout(rollout_eval.cost_vars,ax_rollout)
            setColor(h,i_update,n_updates)
            
         
        ax_space = fig.add_subplot(1,n_subplots,i_subplot)
        i_subplot += 1
        highlight = (i_update==0)
        plotUpdate(distributions,cost_eval,samples,costs,weights,distributions_new,ax_space,highlight)
            
        
    
    try:
        learning_curve = np.loadtxt(directory+'/learning_curve.txt')
    except IOError:
        pass

    axs = [fig.add_subplot(1,n_subplots,i_subplot), fig.add_subplot(1,n_subplots,i_subplot+1)]
    plotCurve(learning_curve,axs)

#def plotOptimizations(directories,axs):
#    n_updates = 10000000
#    for directory in directories:
#        cur_n_updates = loadNumberOfUpdates(directory)
#        n_updates = min([cur_n_updates, n_updates])
#    
#    n_dirs = len(directories)
#    all_costs_eval      = np.empty((n_dirs,n_updates),   dtype=float)
#    all_eval_at_samples = np.empty((n_dirs,n_updates), dtype=float)
#    for dd in range(len(directories)):
#        (update_at_samples, tmp, eval_at_samples, costs_eval) = loadLearningCurve(directories[dd])
#        all_costs_eval[dd]      = costs_eval
#        all_eval_at_samples[dd] = eval_at_samples
#        
#    ax = axs[1]
#    lines_lc = plotLearningCurves(all_eval_at_samples,all_costs_eval,ax)
#    plotUpdateLines(update_at_samples,ax)
#
#    all_sqrt_max_eigvals = np.empty((n_dirs,n_updates+1), dtype=float)
#    all_covar_at_samples = np.empty((n_dirs,n_updates+1), dtype=float)
#    for dd in range(len(directories)):
#        (covar_at_samples, sqrt_max_eigvals) = loadExplorationCurve(directories[dd])
#        all_sqrt_max_eigvals[dd] = sqrt_max_eigvals[:n_updates+1]
#        all_covar_at_samples[dd] = covar_at_samples[:n_updates+1]
#        
#    ax = axs[0]
#    lines_ec = plotExplorationCurves(all_covar_at_samples,all_sqrt_max_eigvals,ax)
#    plotUpdateLines(update_at_samples,ax)
#    
#return (lines_ec, lines_lc)

def saveUpdateRollouts(directory, i_update, distributions, rollout_eval, rollouts, weights, distributions_new):
    
    update_dir = '%s/update%05d' % (directory, i_update)
    if not os.path.exists(update_dir):
        os.makedirs(update_dir)
        
    # Save rollouts
    n_rollouts = len(rollouts)
    for i_rollout in range(n_rollouts):
         cur_dir = '%s/rollout%03d' % (update_dir, i_rollout+1)
         rollouts[i_rollout].saveToDirectory(cur_dir)

    if rollout_eval:
        rollout_eval.saveToDirectory(update_dir+'/rollout_eval')
        
    # use saveUpdate to save the rest
    cost_eval = None
    if rollout_eval:
        cost_eval = rollout_eval.total_cost()
        
    if rollouts[0].total_cost():
        costs = [rollout.total_cost() for rollout in rollouts]
    else:
        costs = None
    
    n_dims = len(rollouts[0].policy_parameters)
    samples = np.full((n_rollouts,n_dims),0.0)
    for i_rollout in range(n_rollouts):
        samples[i_rollout,:] = rollouts[i_rollout].policy_parameters
    
    saveUpdate(directory, i_update, distributions, cost_eval, samples, costs, weights, distributions_new)


