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
from collections import OrderedDict

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)


from dmp_bbo.Rollout import Rollout
from dmp_bbo.LearningSessionTask import *
  
  
def runOptimizationTask(task, task_solver, initial_distribution, updater, n_updates, n_samples_per_update,directory):
    
    session = LearningSessionTask(task,initial_distribution,n_samples_per_update,directory)
    
    distribution = initial_distribution
    
    # Optimization loop
    for i_update in range(n_updates):
        print(f'Update: {i_update}')
        
        # 0. Get cost of current distribution mean
        cost_vars_eval = task_solver.performRollout(distribution.mean)
        cost_eval = task.evaluateRollout(cost_vars_eval,distribution.mean)
        
        # 1. Sample from distribution
        samples = distribution.generateSamples(n_samples_per_update)
    
        # Bookkeeping
        if directory:
            session.save(distribution,"distribution",i_update)
            session.save(cost_vars_eval,"cost_vars",i_update,"eval")
            session.save(cost_eval,"cost_eval",i_update,"eval")
            
        # 2. Evaluate the samples
        costs = np.full(n_samples_per_update,0.0)
        for i_sample, sample in enumerate(samples):
            
            # 2A. Perform the rollouts
            cost_vars = task_solver.performRollout(sample)
      
            # 2B. Evaluate the rollouts
            cur_costs = task.evaluateRollout(cost_vars,sample)
            costs[i_sample] = cur_costs[0]
            
            # Bookkeeping
            session.save(sample,"sample",i_update,i_sample)
            session.save(cost_vars,"cost_vars",i_update,i_sample)
            session.save(cur_costs,"cost",i_update,i_sample)
      
        # 3. Update parameters
        distribution, weights = updater.updateDistribution(distribution, samples, costs)
        
        # Bookkeeping
        session.save(samples,'samples',i_update)
        session.save(costs,'costs',i_update)
        session.save(weights,'weights',i_update)
        session.save(distribution,'distribution_new',i_update)
        
        # Plot summary of this update
        #if fig:
        #    highlight = (i_update==0)
        #    #plotUpdate(distribution,cost_eval,samples,costs,weights,distribution_new,ax_space,highlight)
        #    # Try plotting rollout with task solver
        #    h = task_solver.plotRollout(rollout_eval.cost_vars,ax_rollout)
        #    if not h:
        #        # Task solver didn't know how to plot the rollout, try plotting 
        #        # it with task instead
        #        h = task.plotRollout(rollout_eval.cost_vars,ax_rollout)
        #    if h:
        #        # If there is a handle, change its color depending on i_update
        #        setColor(h,i_update,n_updates)
            
    
    # Remove duplicate entries in the legend.
    #handles, labels = ax_rollout.get_legend_handles_labels()
    #by_label = OrderedDict(zip(labels, handles))
    #ax_rollout.legend(by_label.values(), by_label.keys())
        
    return session
