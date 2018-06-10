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
import math
import os
import sys
import matplotlib.pyplot as plt

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)


from bbo.bbo_plotting import plotUpdate, plotExplorationCurve, plotLearningCurve 
from bbo.bbo_plotting import saveUpdate, saveExplorationCurve, saveLearningCurve

from dmp_bbo.dmp_bbo_plotting import saveUpdateRollouts, setColor
from dmp_bbo.Rollout import Rollout
  
  
def runOptimizationTask(task, task_solver, initial_distribution, updater, n_updates, n_samples_per_update,fig=None,directory=None):
    
    distribution = initial_distribution
    
    all_costs = []
    learning_curve = []
    exploration_curve = []
    
    if fig:
        ax_space   = fig.add_subplot(141)
        ax_rollout = fig.add_subplot(142)
    
    # Optimization loop
    for i_update in range(n_updates): 
        
        # 0. Get cost of current distribution mean
        cost_vars_eval = task_solver.performRollout(distribution.mean)
        cost_eval = task.evaluateRollout(cost_vars_eval,distribution.mean)
        rollout_eval = Rollout(distribution.mean,cost_vars_eval,cost_eval)
        
        # 1. Sample from distribution
        samples = distribution.generateSamples(n_samples_per_update)
    
        # 2. Evaluate the samples
        costs = np.full(n_samples_per_update,0.0)
        rollouts = []
        for i_sample in range(n_samples_per_update):
            
            # 2A. Perform the rollouts
            cost_vars = task_solver.performRollout(samples[i_sample,:])
      
            # 2B. Evaluate the rollouts
            cur_costs = task.evaluateRollout(cost_vars,samples[i_sample,:])
            costs[i_sample] = cur_costs[0]

            rollouts.append(Rollout(samples[i_sample,:],cost_vars,cur_costs))
      
        # 3. Update parameters
        distribution_new, weights = updater.updateDistribution(distribution, samples, costs)
        
        # Bookkeeping and plotting
        # All the costs so far
        all_costs.extend(costs)
        # Update exploration curve
        cur_samples = i_update*n_samples_per_update
        cur_exploration = np.sqrt(distribution.maxEigenValue())
        exploration_curve.append([cur_samples,cur_exploration])
        # Update learning curve
        learning_curve.append([cur_samples])
        learning_curve[-1].extend(cost_eval)
        
        
        # Plot summary of this update
        if fig:
            highlight = (i_update==0)
            plotUpdate(distribution,cost_eval,samples,costs,weights,distribution_new,ax_space,highlight)
            # Try plotting rollout with task solver
            h = task_solver.plotRollout(rollout_eval.cost_vars,ax_rollout)
            if not h:
                # Task solver didn't know how to plot the rollout, try plotting 
                # it with task instead
                h = task.plotRollout(rollout_eval.cost_vars,ax_rollout)
            if h:
                # If there is a handle, change its color depending on i_update
                setColor(h,i_update,n_updates)
            
        if directory:
            saveUpdateRollouts(directory, i_update, distribution, rollout_eval, rollouts, weights, distribution_new)
        
        # Distribution is new distribution
        distribution = distribution_new
        
        
    # Plot learning curve
    cost_labels = task.costLabels()
    if fig:
        plotExplorationCurve(exploration_curve,fig.add_subplot(143))
        plotLearningCurve(learning_curve,fig.add_subplot(144),all_costs,cost_labels)

    # Save learning curve to file, if necessary
    if directory:
        saveLearningCurve(directory,learning_curve)
        saveExplorationCurve(directory,exploration_curve)
        print('Saved results to "'+directory+'".')
    
    return learning_curve
