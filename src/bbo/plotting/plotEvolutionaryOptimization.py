import sys
import numpy                                                                    
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import os
import matplotlib.pyplot as pl
import time
#from matplotlib import animation

from plotUpdateSummary import plotUpdateSummaryFromDirectory


def plotUpdateLines(n_samples_per_update,ax,y_limits=[]):
    if (len(y_limits)==0):
        y_limits = ax.get_ylim()
    
    # Find good number of horizontal update lines to plot    
    updates = numpy.arange(0, len(n_samples_per_update))
    while len(n_samples_per_update)>20:
      n_samples_per_update = n_samples_per_update[0:-1:5]
      updates = updates[0:-1:5]
    
    ax.plot([n_samples_per_update, n_samples_per_update],y_limits,'-',color='#bbbbbb',linewidth=0.5,zorder=0)
    for ii in range(len(n_samples_per_update)-1):
        y = y_limits[0] + 0.9*(y_limits[1]-y_limits[0])
        ax.text(n_samples_per_update[ii+1], y,str(updates[ii+1]),  
                          horizontalalignment='center',verticalalignment='top',rotation='vertical')
        
    y = y_limits[0] + 0.95*(y_limits[1]-y_limits[0])
    ax.text(mean(ax.get_xlim()), y,'number of updates',
        horizontalalignment='center', verticalalignment='top')
    ax.set_ylim(y_limits)

def loadNumberOfUpdates(directory):
    n_updates = 0;
    dir_exists = True;
    while (dir_exists):
      n_updates+=1
      cur_directory = '%s/update%05d' % (directory, n_updates)
      dir_exists = os.path.isdir(cur_directory)
    n_updates-=1
    return n_updates

def loadLearningCurve(directory):
    n_updates = loadNumberOfUpdates(directory)
    
    costs_all = []
    costs_eval = []
    update_at_samples = []
    has_noise_free_eval = False;
    for update in range(n_updates):
        cur_directory = '%s/update%05d' % (directory, update+1)
        # Load costs of individual samples
        cur_costs = np.loadtxt(cur_directory+"/costs.txt")
        costs_all.extend(cur_costs)
        update_at_samples.append(len(costs_all))   
        try:
            # Load evaluation cost, if it exists
            cur_cost_eval = np.loadtxt(cur_directory+"/cost_eval.txt")
            costs_eval.append(np.atleast_1d(cur_cost_eval)[0])
            has_noise_free_eval = True
        except IOError:
            # If there was no noise-free evaluation, take the mean of this epoch
            costs_eval.append(mean(cur_costs))

    if (has_noise_free_eval):
        # The noise-free evaluations are done exactly at the updates
        eval_at_samples = [0];
        eval_at_samples.extend(update_at_samples[:-1])
    else:
        # The means are centered between two updates
        n = [0]
        n.extend(update_at_samples)
        eval_at_samples = [0.5*(n[i] + n[i+1]) for i in range(len(n)-1)]
        
    return (update_at_samples, costs_all, eval_at_samples, costs_eval)

def loadExplorationCurve(directory,subdir=""):
    n_updates = loadNumberOfUpdates(directory)

    # Load all the covar matrices
    covar_at_samples = [0]
    covars_per_update = [];
    for update in range(n_updates):
        cur_directory = '%s/update%05d/' % (directory, update+1)
        covar = np.loadtxt(cur_directory+subdir+"/distribution_covar.txt")
        covars_per_update.append(covar)
        cur_costs = np.loadtxt(cur_directory+"/costs.txt")
        covar_at_samples.append(covar_at_samples[-1]+len(cur_costs))

    # Load final covar matrix
    covar = np.loadtxt(cur_directory+subdir+"/distribution_new_covar.txt")
    covars_per_update.append(covar)
        
    # Compute sqrt of max of eigenvalues
    sqrt_max_eigvals = [];
    for update in range(len(covars_per_update)):
        if (isnan(covars_per_update[update]).any()):
            print update
            print covars_per_update[update]
            print "Found nan in covariance matrix..."
        eigvals, eigvecs = np.linalg.eig(covars_per_update[update])
        sqrt_max_eigvals.append(sqrt(max(eigvals)))

    return (covar_at_samples, sqrt_max_eigvals)

def computeMeanCostsPerUpdateDeprecated(n_samples_per_update,costs_per_sample):
    # Take the mean of the costs per update 
    # (first, compute the center of each update)
    n = n_samples_per_update
    centers_n_samples_per_update = [0.5*(n[i] + n[i+1]) for i in range(len(n)-1)]
    mean_samples_per_update = [ np.mean(costs_per_sample[n[i]:n[i+1]]) for i in range(len(n)-1)]
    return (centers_n_samples_per_update, mean_samples_per_update)
    

def plotLearningCurves(all_eval_at_samples,all_costs_eval,ax):
    
    # Check if all n_samples_per_update are the same. Otherwise averaging doesn't make sense.
    std_samples = all_eval_at_samples.std(0)
    if (sum(std_samples)>0.0001):
        print "WARNING: updates must be the same"
    eval_at_samples = all_eval_at_samples[0];
            
    # Compute average and standard deviation for learning and exploration curves
    mean_costs = all_costs_eval.mean(0)
    std_costs  = all_costs_eval.std(0)
        
    # Plot costs of all individual samples 
    line_mean = ax.plot(eval_at_samples,mean_costs,'-',color='blue',linewidth=2)
    line_std_plus = ax.plot(eval_at_samples,mean_costs+std_costs,'-',color='blue',linewidth=1)
    line_std_min = ax.plot(eval_at_samples,mean_costs-std_costs,'-',color='blue',linewidth=1)
    ax.set_xlabel('number of evaluations')
    ax.set_ylabel('cost')
    ax.set_title('Learning curve')
    return (line_mean, line_std_plus, line_std_min)

def plotLearningCurve(samples_eval,costs_eval,ax,costs_all=[]):
    # Plot costs of all individual samples 
    if (len(costs_all)>0):
        ax.plot(costs_all,'.',color='gray')
    # Plot costs at evaluations 
    line = ax.plot(samples_eval,costs_eval,'-',color='blue',linewidth=2)
    ax.set_xlabel('number of evaluations')
    ax.set_ylabel('cost')
    ax.set_title('Learning curve')
    y_limits = [0,1.2*max(costs_eval)];
    ax.set_ylim(y_limits)
    return line
      
def plotExplorationCurves(all_covar_at_samples,all_exploration_curves,ax):
    # Check if all n_samples_per_update are the same. Otherwise averaging doesn't make sense.
    std_samples = all_covar_at_samples.std(0)
    if (sum(std_samples)>0.0001):
        print "WARNING: updates must be the same"
    covar_at_samples = all_covar_at_samples[0];

    # Compute average and standard deviation for learning and exploration curves
    mean_curve = all_exploration_curves.mean(0)
    std_curve  = all_exploration_curves.std(0)

    x = covar_at_samples
    line_mean     = ax.plot(x,mean_curve,'-',color='green',linewidth=2)
    line_std_plus = ax.plot(x,mean_curve+std_curve,'-',color='green',linewidth=1)
    line_std_min  = ax.plot(x,mean_curve-std_curve,'-',color='green',linewidth=1)
    ax.set_xlabel('number of evaluations')
    ax.set_ylabel('sqrt of max. eigval of covar')
    ax.set_title('Exploration magnitude')
    return (line_mean, line_std_plus, line_std_min)
    
def plotExplorationCurve(n_samples_per_update,exploration_curve,ax):
    line = ax.plot(n_samples_per_update,exploration_curve,'-',color='green',linewidth=2)
    ax.set_xlabel('number of evaluations')
    ax.set_ylabel('sqrt of max. eigval of covar')
    ax.set_title('Exploration magnitude')
    return line

def plotEvolutionaryOptimization(directory,axs,plot_all_rollouts=False):
      
    #################################
    # Load and plot learning curve
    (update_at_samples, costs_all, eval_at_samples, costs_eval) = loadLearningCurve(directory)
    ax = (None if axs==None else axs[1])
    plotLearningCurve(eval_at_samples,costs_eval,ax,costs_all)
    #y_limits = [0,1.2*max(learning_curve)];
    plotUpdateLines(update_at_samples,ax)
    
    
    #################################
    # Load and plot exploration curve
    (covar_at_samples, sqrt_max_eigvals) = loadExplorationCurve(directory)
    ax = (None if axs==None else axs[0])
    plotExplorationCurve(covar_at_samples,sqrt_max_eigvals,ax)
    plotUpdateLines(update_at_samples,ax)

    n_updates = loadNumberOfUpdates(directory)
    ax = (None if axs==None else axs[2])
    if (ax!=None):
        #################################
        # Visualize the update in parameter space 
        for update in range(n_updates):
            cur_directory = '%s/update%05d' % (directory, update+1)
            plotUpdateSummaryFromDirectory(cur_directory,ax,False)
            
        cur_directory = '%s/update%05d' % (directory, n_updates)
        plotUpdateSummaryFromDirectory(cur_directory,ax,True)
        ax.set_title('Search space')
        
    if ( (axs!=None) and (len(axs)>3) ):
        ax = axs[3];
        rollouts_python_script = directory+'/plotRollouts.py'
        if (os.path.isfile(rollouts_python_script)):
            lib_path = os.path.abspath(directory)
            sys.path.append(lib_path)
            from plotRollouts import plotRollouts
        
            for update in range(n_updates):
                cur_directory = '%s/update%05d' % (directory, update+1)
                filename = "/cost_vars_eval.txt"
                if plot_all_rollouts:
                    filename = "/cost_vars.txt"
                cost_vars = np.loadtxt(cur_directory+filename)
                rollout_lines = plotRollouts(cost_vars,ax)
                color_val = (1.0*update/n_updates)
                #cur_color = [1.0-0.9*color_val,0.1+0.9*color_val,0.1]
                cur_color = [0.0-0.0*color_val, 0.0+1.0*color_val, 0.0-0.0*color_val]
                #print str(update)+" "+str(n_updates)+" "+str(cur_color) 
                plt.setp(rollout_lines,color=cur_color)
                if (update==0):
                    plt.setp(rollout_lines,color='r',linewidth=2)

def plotEvolutionaryOptimizations(directories,axs):
    n_updates = 10000000
    for directory in directories:
        cur_n_updates = loadNumberOfUpdates(directory)
        n_updates = min([cur_n_updates, n_updates])
    
    n_dirs = len(directories)
    all_costs_eval      = np.empty((n_dirs,n_updates),   dtype=float)
    all_eval_at_samples = np.empty((n_dirs,n_updates), dtype=float)
    for dd in range(len(directories)):
        (update_at_samples, tmp, eval_at_samples, costs_eval) = loadLearningCurve(directories[dd])
        all_costs_eval[dd]      = costs_eval
        all_eval_at_samples[dd] = eval_at_samples
        
    ax = axs[1]
    lines_lc = plotLearningCurves(all_eval_at_samples,all_costs_eval,ax)
    plotUpdateLines(update_at_samples,ax)

    all_sqrt_max_eigvals = np.empty((n_dirs,n_updates+1), dtype=float)
    all_covar_at_samples = np.empty((n_dirs,n_updates+1), dtype=float)
    for dd in range(len(directories)):
        (covar_at_samples, sqrt_max_eigvals) = loadExplorationCurve(directories[dd])
        all_sqrt_max_eigvals[dd] = sqrt_max_eigvals[:n_updates+1]
        all_covar_at_samples[dd] = covar_at_samples[:n_updates+1]
        
    ax = axs[0]
    lines_ec = plotExplorationCurves(all_covar_at_samples,all_sqrt_max_eigvals,ax)
    plotUpdateLines(update_at_samples,ax)
    
    return (lines_ec, lines_lc)

    
if __name__=='__main__':
    
    # See if input directory was passed
    if (len(sys.argv)<2):
        print '\nUsage: '+sys.argv[0]+' <directory> [directory2] [directory3] [etc]\n';
        sys.exit()
        
    directories = []
    for arg in sys.argv[1:]:
        directories.append(str(arg))
    
    
    if (len(directories)==1):
        rollouts_python_script = directories[0]+'/plotRollouts.py'
        if (os.path.isfile(rollouts_python_script)):
            lib_path = os.path.abspath(directories[0])
            sys.path.append(lib_path)
            from plotRollouts import plotRollouts
            fig = plt.figure(1,figsize=(12, 4))
            axs = [ fig.add_subplot(143), fig.add_subplot(144), fig.add_subplot(142) , fig.add_subplot(141) ]
        else:
            fig = plt.figure(1,figsize=(9, 4))
            axs = [ fig.add_subplot(132), fig.add_subplot(133), fig.add_subplot(131) ]
        plotEvolutionaryOptimization(directories[0],axs)
    else:
        fig = plt.figure(1,figsize=(6, 4))
        axs = [ fig.add_subplot(121),  fig.add_subplot(122) ]
        plotEvolutionaryOptimizations(directories,axs)
 
    plt.show()

