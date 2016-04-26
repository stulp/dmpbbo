#ZZZ REMOVE UNCESSESARY IMPORTS
import sys
import numpy                                                                    
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import os
import matplotlib.pyplot as pl
import time
#from matplotlib import animation

def loadNumberOfUpdates(directory):
    n_updates = 0;
    dir_exists = True;
    while (dir_exists):
      n_updates+=1
      cur_directory = '%s/update%05d' % (directory, n_updates)
      dir_exists = os.path.isdir(cur_directory)
      #if (dir_exists):
      #    # File distribution_new_mean.txt must exist also. Otherwise
      #    # the update didn't really happen.
      #    if not os.path.isfile(cur_directory+'/distribution_new_mean.txt'):
      #        dir_exists = False
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

def loadExplorationCurve(directory,i_parallel=-1):
    n_updates = loadNumberOfUpdates(directory)

    suffix=""
    if (i_parallel>=0):
        suffix = '_%02d' % i_parallel

    # Load all the covar matrices
    covar_at_samples = [0]
    covars_per_update = [];
    for update in range(n_updates):
        cur_directory = '%s/update%05d/' % (directory, update+1)
        covar = np.loadtxt(cur_directory+"/distribution_covar"+suffix+".txt")
        covars_per_update.append(covar)
        cur_costs = np.loadtxt(cur_directory+"/costs.txt")
        covar_at_samples.append(covar_at_samples[-1]+len(cur_costs))

    # Load final covar matrix
    covar = np.loadtxt(cur_directory+"/distribution_new_covar"+suffix+".txt")
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

def plotLearningCurveDir(directory,ax):
    learning_curve = np.loadtxt(directory+'/learning_curve.txt')
    plotLearningCurve(learning_curve[:,0],learning_curve[:,1],ax)
    plotUpdateLines(learning_curve[:,0],ax)
    
def plotExplorationCurveDir(directory,ax):
    learning_curve = np.loadtxt(directory+'/learning_curve.txt')
    if learning_curve.shape[1]>2:
        plotLearningCurve(learning_curve[:,0],learning_curve[:,2],ax)
        plotUpdateLines(learning_curve[:,0],ax)
    

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
    
    
def plotUpdateSummary(update_summary,ax,highlight=True,plot_samples=False):
    us = update_summary
    
    if us.samples==None:
        plot_samples = False
    
    n_parallel = len(us.distributions)
    n_dims = len(us.distributions[0].mean)
    if (n_dims==1):
        print "Sorry, only know how to plot for n_dims==2, but you provided n_dims==1"
        return
    for i_parallel in range(n_parallel):
        # ZZZ Perhaps plot in different subplots
        if (n_dims>=2):
            #print "Sorry, only know how to plot for n_dims==2, throwing away excess dimensions"
            distr_mean = us.distributions[i_parallel].mean[0:2]
            distr_covar = us.distributions[i_parallel].covar[0:2,0:2]
            distr_new_mean  = us.distributions_new[i_parallel].mean[0:2]
            distr_new_covar = us.distributions_new[i_parallel].covar[0:2,0:2]
            if us.samples!=None:
                samples = us.samples[i_parallel,0:2]
                    
        if plot_samples:
            max_marker_size = 80;
            for ii in range(len(weights)):
                cur_marker_size = max_marker_size*weights[ii]
                sample_handle = ax.plot(samples[ii,0],samples[ii,1],'o',color='green')
                plt.setp(sample_handle,markersize=cur_marker_size,markerfacecolor=(0.5,0.8,0.5),markeredgecolor='none')
          
                ax.plot(samples[:,0],samples[:,1],'.',color='black')
            ax.plot((distr_mean[0],distr_new_mean[0]),(distr_mean[1],distr_new_mean[1]),'-',color='blue')
            
        mean_handle = ax.plot(distr_mean[0],distr_mean[1],'o',label='old')
        mean_handle_new = ax.plot(distr_new_mean[0],distr_new_mean[1],'o',label='new')
        mean_handle_link = ax.plot([distr_mean[0], distr_new_mean[0]],[distr_mean[1], distr_new_mean[1]],'-')
        patch = plot_error_ellipse(distr_mean[0:2],distr_covar[0:2,0:2],ax)
        patch_new = plot_error_ellipse(distr_new_mean[0:2],distr_new_covar[0:2,0:2],ax)
        if (highlight):
            plt.setp(mean_handle,color='red')
            plt.setp(mean_handle_new,color='blue')
            plt.setp(patch,edgecolor='red')
            plt.setp(patch_new,edgecolor='blue')
        else:
            plt.setp(mean_handle,color='gray')
            plt.setp(mean_handle_new,color='gray')
            plt.setp(patch,edgecolor='gray')
            plt.setp(patch_new,edgecolor='gray')
        plt.setp(mean_handle_link,color='gray')
    ax.set_aspect('equal')

    plt.rcParams['text.usetex']=True
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    return mean_handle,mean_handle_new,patch,patch_new
    

#if __name__=='__main__':
#    
#    # See if input directory was passed
#    if (len(sys.argv)==2):
#      directory = str(sys.argv[1])
#    else:
#      print '\nUsage: '+sys.argv[0]+' <directory>\n';
#      sys.exit()
#    
#    update_summary = loadFromDirectory(directory)
#    
#    fig = plt.figure()
#    ax = fig.gca()
#    highlight = True
#    plot_samples = True
#    plotUpdateSummaryFromDirectory(update_summary,ax,highlight,plot_samples)
#    plt.show()
    
    
def plotEvolutionaryOptimizationDir(directory,axs=None,plot_all_rollouts=False):
    if axs==None:
        fig = plt.figure(1,figsize=(9, 4))
        axs = [ fig.add_subplot(132), fig.add_subplot(133), fig.add_subplot(131) ]
    
    n_updates = loadNumberOfUpdates(directory)
    update_summaries = []
    for i_update in range(1,n_updates):
        cur_directory = '%s/update%05d' % (directory, i_update)
        us = loadFromDirectory(cur_directory)
        update_summaries.append(us)
        
    plotRollout = None
    rollouts_python_script = directory+'/plotRollout.py'
    if (os.path.isfile(rollouts_python_script)):
        lib_path = os.path.abspath(directory)
        sys.path.append(lib_path)
        from plotRollout import plotRollout
   
    return plotEvolutionaryOptimization(update_summaries,axs,plotRollout,plot_all_rollouts)
        

def plotEvolutionaryOptimization(update_summaries,axs,plotRolloutFunc=None,plot_all_rollouts=False):

    curves = extractLearningCurve(update_summaries)
    samples_eval = curves[:,0]
    costs_eval   = curves[:,1]
    explo_curve  = curves[:,2]
    costs_all = []
    for us in update_summaries:
        costs_all.extend(us.costs)
    
    ax = (None if axs==None else axs[0])
    if ax:
        plotExplorationCurve(samples_eval,explo_curve,ax)
        plotUpdateLines(samples_eval,ax)

    ax = (None if axs==None else axs[1])
    if ax:
        plotLearningCurve(samples_eval,costs_eval,ax,costs_all)
        plotUpdateLines(samples_eval,ax)

    n_updates = len(update_summaries)
    ax = (None if axs==None else axs[2])
    if (ax!=None):
        #################################
        # Visualize the update in parameter space 
        for update_summary in update_summaries:
            plotUpdateSummary(update_summary,ax,False)
        plotUpdateSummary(update_summaries[-1],ax,True)
        ax.set_title('Search space')
        
    if (plotRolloutFunc and (axs!=None) and (len(axs)>3)):
        ax = axs[3];
    
        for update in range(n_updates):
            cost_vars = [update_summaries[update].cost_vars_eval]
            if plot_all_rollouts:
                cost_vars = update_summaries[update].cost_vars
                
            for i_rollout in range(len(cost_vars)):
                cur_cost_vars = cost_vars[i_rollout]

                rollout_lines = plotRolloutFunc(cur_cost_vars,ax)
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
        rollouts_python_script = directories[0]+'/plotRollout.py'
        if (os.path.isfile(rollouts_python_script)):
            lib_path = os.path.abspath(directories[0])
            sys.path.append(lib_path)
            from plotRollout import plotRollout
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

def saveUpdateRollouts(directory, i_update, distributions, rollout_eval, rollouts, weights, distributions_new):
    
    cost_eval = None
    if rollout_eval:
        cost_eval = rollout_eval.total_cost()
    
    costs = [rollout.total_cost() for rollout in rollouts]

    samples = None
    saveUpdate(directory, i_update, distributions, cost_eval, samples, costs, weights, distributions_new);

    update_dir = '%s/update%05d' % (directory, i_update)
    if not os.path.exists(update_dir):
        os.makedirs(update_dir)
        
    # Save rollouts too
    n_rollouts = len(rollouts)
    for i_rollout in range(n_rollouts):
         cur_dir = '%s/rollout%03d' % (update_dir, i_rollout+1)
         rollouts[i_rollout].saveToDirectory(cur_dir)

    if rollout_eval:
        rollout_eval.saveToDirectory(update_dir+'/rollout_eval')

