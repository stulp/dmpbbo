import numpy as np
import math
import os
import sys

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from rollout import loadRolloutFromDirectory, loadRolloutsFromDirectory 
from dmp_bbo_plotting import saveUpdateRollouts
from bbo.distribution_gaussian import DistributionGaussian
from bbo.distribution_gaussian import loadDistributionGaussianFromDirectory

def runOptimizationTaskOneUpdate(task, initial_distribution, updater, n_samples_per_update, directory):

    # If directory doesnt exist
    if not os.path.isdir(directory):
        os.makedirs(directory)
        i_update = 0
    else:
        i_update = 0;
        dir_exists = True;
        while (dir_exists):
          i_update +=1
          cur_directory = '%s/update%05d' % (directory, i_update)
          dir_exists = os.path.isdir(cur_directory)
        i_update-=1
        
    update_dir = '%s/update%05d' % (directory, i_update)
    if not os.path.isdir(update_dir):
        os.makedirs(update_dir)
        
        
    print('======================================================')
    print('i_update = '+str(i_update))
    if i_update==0:
        print('INITIALIZING')
        print('  * Save distribution to "'+update_dir+'/"')
        # First update: current distribution is initial distribution
        distribution_new = initial_distribution
        
    else:
        
        
        print('EVALUATING')
        
        rollout_eval = loadRolloutFromDirectory(update_dir+'/rollout_eval')
        rollouts = loadRolloutsFromDirectory(update_dir)
        assert(len(rollouts)==n_samples_per_update)
        assert(rollouts[0])
        
        print('  * Evaluating costs')
        costs = np.full(n_samples_per_update,0.0)
        for i_rollout in range(len(rollouts)):
          
            # 2B. Evaluate the samples
            cur_costs = task.evaluateRollout(rollouts[i_rollout].cost_vars)
            rollouts[i_rollout].cost = cur_costs
            costs[i_rollout] = np.atleast_1d(cur_costs)
      
        # 3. Update parameters
        print('UPDATING')
        print('  * Loading distribution, samples and cost_vars from  "'+update_dir+'/"')
        name = 'distribution'
        distribution = loadDistributionGaussianFromDirectory(update_dir,name)
        
        samples = np.loadtxt(update_dir+"/samples.txt")
        
        print('  * Updating parameters')
        distribution_new, weights = updater.updateDistribution(distribution, samples, costs)
        
        # Save this update to file
        print('  * Saving update to  "'+update_dir+'/"')
        saveUpdateRollouts(directory, i_update, distribution, rollout_eval, rollouts, weights, distribution_new)
        
    print('  * Saving updated distribution to "'+update_dir+'/"')
    np.savetxt(update_dir+"/distribution_new_mean.txt",distribution_new.mean)
    np.savetxt(update_dir+"/distribution_new_covar.txt",distribution_new.covar)
    
    # Update done! Increment counter, and save distribution to new dir.
    i_update += 1
    update_dir = '%s/update%05d' % (directory, i_update)
    # If directory doesnt exist
    if not os.path.isdir(update_dir):
        os.makedirs(update_dir)
    print('  * Saving updated distribution to "'+update_dir+'/"')
    np.savetxt(update_dir+"/distribution_mean.txt",distribution_new.mean)
    np.savetxt(update_dir+"/distribution_covar.txt",distribution_new.covar)
        
    # 1. Sample from distribution (for next epoch of rollouts)
    print('SAMPLING')
    samples = distribution_new.generateSamples(n_samples_per_update)
    print('  * Save samples to "'+update_dir+'/samples.txt"')
    np.savetxt(update_dir+"/samples.txt",samples)
    
    print('ROLLOUTS')
    print('  * Info: '+str(n_samples_per_update)+' samples have been save in "'+update_dir+'"/samples.txt".')
    print('    Please run '+str(n_samples_per_update)+' rollouts on the robot and write cost-relevant variables in "'+directory+'/cost_vars.txt"')

    return i_update




