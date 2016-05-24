import numpy as np
import math
import os
import sys
import pickle
import inspect

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from dmp_bbo.rollout import loadRolloutFromDirectory, loadRolloutsFromDirectory 
from dmp_bbo.dmp_bbo_plotting import saveUpdateRollouts
from bbo.distribution_gaussian import DistributionGaussian
from bbo.distribution_gaussian import loadDistributionGaussianFromDirectory
from dmp_bbo.task import Task

def prepareOptimization(directory,task,initial_distribution,updater,    n_samples_per_update):
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    print('  * Saving task to "'+directory+'/"')
    # Save the source code of the task for future reference
    src_task = inspect.getsourcelines(task.__class__)
    src_task = ' '.join(src_task[0])
    src_task = src_task.replace("(Task)", "")    
    filename = directory+'/the_task.py'
    task_file = open(filename, "w")
    task_file.write(src_task)
    task_file.close()
    
    # Save the task instance itself
    filename = directory+'/task.p'
    pickle.dump(task, open(filename, "wb" ))

    print('  * Saving updater to "'+directory+'/"')
    filename = directory+'/updater.p'
    pickle.dump(updater, open(filename, "wb" ))

    filename = directory+'/n_samples_per_update.txt'
    np.savetxt(filename,[n_samples_per_update])
    
    print('  * Saving initial distribution to "'+directory+'/"')
    initial_distribution.saveToDirectory(directory,'distribution_initial')
    
    
def runOptimizationTaskOneUpdate(directory,task, initial_distribution, updater, n_samples_per_update,i_update=None):

    if not i_update:
        i_update = -1;
        dir_exists = True;
        while (dir_exists):
          i_update +=1
          cur_directory = '%s/update%05d/rollout001' % (directory, i_update)
          dir_exists = os.path.isdir(cur_directory)
        i_update-=1
        
    if i_update>=0:
        update_dir = '%s/update%05d' % (directory, i_update)
        if not os.path.isdir(update_dir):
            os.makedirs(update_dir)
        
    print('======================================================')
    print('i_update = '+str(i_update))
    if i_update==-1:
        print('NO UPDATES YET: PREPARE OPTIMIZATION')
        
        prepareOptimization(directory,task,initial_distribution,updater,    n_samples_per_update)
            
        # First update: distribution is initial distribution
        distribution_new = initial_distribution
        
    else:
        
        print('EVALUATING ROLLOUTS')
        
        print('  * Loading rollouts')
        rollout_eval = loadRolloutFromDirectory(update_dir+'/rollout_eval')
        rollouts = loadRolloutsFromDirectory(update_dir)
        n_samples = len(rollouts)
    
        print('  * Evaluating costs')
        sample = rollout_eval.policy_parameters
        cost_eval = task.evaluateRollout(rollout_eval.cost_vars,sample)
        rollout_eval.cost = cost_eval
        
        costs = []
        for i_rollout in range(n_samples):
          
            # 2B. Evaluate the samples
            cost_vars = rollouts[i_rollout].cost_vars
            sample = rollouts[i_rollout].policy_parameters
            cur_costs = task.evaluateRollout(cost_vars,sample)
            rollouts[i_rollout].cost = cur_costs
            costs.append(cur_costs)
      
        # 3. Update parameters
        print('UPDATING DISTRIBUTION')
        print('  * Loading previous distribution and samples from  "'+update_dir+'/"')
        name = 'distribution'
        distribution = loadDistributionGaussianFromDirectory(update_dir,name)
        
        samples = np.loadtxt(update_dir+"/samples.txt")

        print('  * Loading updater')
        updater = pickle.load(open(directory+'/updater.p', "rb" ))

        print('  * Updating parameters')
        distribution_new, weights = updater.updateDistribution(distribution, samples, costs)
        
        # Save this update to file
        print('  * Saving update to  "'+update_dir+'/"')
        saveUpdateRollouts(directory, i_update, distribution, rollout_eval, rollouts, weights, distribution_new)
        
        print('  * Saving distribution to "'+update_dir+'/"')
        np.savetxt(update_dir+"/distribution_new_mean.txt",distribution_new.mean)
        np.savetxt(update_dir+"/distribution_new_covar.txt",distribution_new.covar)
    
    # Update done! Increment counter, and save distribution to new dir.
    i_update += 1
    update_dir = '%s/update%05d' % (directory, i_update)
    # If directory doesnt exist
    if not os.path.isdir(update_dir):
        os.makedirs(update_dir)
    print('  * Saving distribution to "'+update_dir+'/"')
    np.savetxt(update_dir+"/distribution_mean.txt",distribution_new.mean)
    np.savetxt(update_dir+"/distribution_covar.txt",distribution_new.covar)
        
    # 1. Sample from distribution (for next epoch of rollouts)
    print('SAMPLING FROM UPDATED DISTRIBUTION')
    samples = distribution_new.generateSamples(n_samples_per_update)
    print('  * Save samples to "'+update_dir+'/samples.txt"')
    np.savetxt(update_dir+"/samples.txt",samples)
    
    print('ROLLOUTS NOW REQUIRED')
    print('  * Info: '+str(n_samples_per_update)+' samples have been save in "'+update_dir+'/samples.txt".')
    print('    Please run '+str(n_samples_per_update)+' rollouts on the robot and write cost-relevant variables in "'+update_dir+'/cost_vars.txt"')

    return i_update
