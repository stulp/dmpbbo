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


def plotUpdateLines(n_samples_per_update,y_limits,ax):

    ax.plot([n_samples_per_update, n_samples_per_update],y_limits,'-',color='#bbbbbb',linewidth=0.5,zorder=0)
    for ii in range(len(n_samples_per_update)-2):
        y = y_limits[0] + 0.9*(y_limits[1]-y_limits[0])
        if ( (n_samples_per_update<15) | ((ii+1)%5==0) ): 
            ax.text(n_samples_per_update[ii+1], y,str(ii+1),
            horizontalalignment='center',verticalalignment='top',rotation='vertical')
        
    y = y_limits[0] + 0.95*(y_limits[1]-y_limits[0])
    ax.text(mean(ax.get_xlim()), y,'number of updates',
        horizontalalignment='center', verticalalignment='top')


def plotLearningCurve(n_samples_per_update,costs_per_sample,cost_evals=[],ax=None):

    if (len(cost_evals)==0):
        # Take the mean of the costs per update 
        # (first, compute the center of each update)
        n = n_samples_per_update
        centers_n_samples_per_update = [0.5*(n[i] + n[i+1]) for i in range(len(n)-1)]
        mean_samples_per_update = [ np.mean(costs_per_sample[n[i]:n[i+1]]) for i in range(len(n)-1)]

        xs = centers_n_samples_per_update
        ys = mean_samples_per_update
    else:
        xs = n_samples_per_update[0:-1]
        ys = cost_evals
    
    
    if (ax!=None):
        # Plot costs of all individual samples 
        ax.plot(costs_per_sample,'.',color='gray')
        ax.plot(xs,ys,'-',color='blue',linewidth=2)
        y_limits = [0,1.1*max(ys)];
        ax.set_xlabel('number of evaluations')
        ax.set_ylabel('mean cost per update')
        ax.set_title('Learning curve')
        plotUpdateLines(n_samples_per_update,y_limits,ax)
        ax.set_ylim(y_limits)

    return (xs, ys)
  

def plotExplorationCurve(n_samples_per_update,covars_per_update,ax=None):
    max_eigval_per_update = [];
    for update in range(len(covars_per_update)):
        if (isnan(covars_per_update[update]).any()):
            print update
            print covars_per_update[update]
        eigvals, eigvecs = np.linalg.eig(covars_per_update[update])
        max_eigval_per_update.append(max(eigvals))

    if (ax!=None):
        ax.plot(n_samples_per_update,sqrt(max_eigval_per_update),'-',color='green',linewidth=2)
        y_limits = ax.get_ylim()
        plotUpdateLines(n_samples_per_update,y_limits,ax)
        ax.set_xlabel('number of evaluations')
        ax.set_ylabel('sqrt of max. eigval of covar')
        ax.set_title('Exploration magnitude')
        
    return max_eigval_per_update
  

def plotEvolutionaryOptimization(n_updates,directory,axs=None,plot_all_rollouts=False):

    if (n_updates==0):
      n_updates += 1;
      
    #################################
    # Read data relevant to learning and exploration curve
    costs = []
    cost_evals = []
    n_samples_per_update = [0]
    covars_per_update = [];
    for update in range(n_updates):
      
        cur_directory = '%s/update%05d' % (directory, update+1)
        
        # Load evaluation cost
        try:
            cur_cost_eval = np.loadtxt(cur_directory+"/cost_eval.txt")
            cost_evals.append(np.atleast_1d(cur_cost_eval)[0])
        except IOError:
            cur_cost_eval = []
            
            
        # Load costs
        cur_costs = np.loadtxt(cur_directory+"/costs.txt")
        costs.extend(cur_costs)
        
        n_samples_per_update.append(len(costs))
        
        # Load covar matrix
        covar = np.loadtxt(cur_directory+"/distribution_covar.txt")
        covars_per_update.append(covar)

    # Load final covar matrix
    covar = np.loadtxt(cur_directory+"/distribution_new_covar.txt")
    covars_per_update.append(covar)
        
    #################################
    # Plot the exploration magnitude
    ax = (None if axs==None else axs[0])
    max_eigval_per_update = plotExplorationCurve(n_samples_per_update,covars_per_update,ax)
    
    #################################
    # Plot the learning curve
    ax = (None if axs==None else axs[1])
    update_centers, mean_costs = plotLearningCurve(n_samples_per_update,costs,cost_evals,ax)

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
                

    return (update_centers, mean_costs, n_samples_per_update, max_eigval_per_update)
    

def plotEvolutionaryOptimizations(n_updates,directories,axs):
    
    all_updates = np.empty((len(directories),n_updates), dtype=float)
    all_mean_costs = np.empty((len(directories),n_updates), dtype=float)
    all_n_samples_per_update = np.empty((len(directories),n_updates+1), dtype=float)
    all_max_eigval_per_update = np.empty((len(directories),n_updates+1), dtype=float)
    for dd in range(len(directories)):

        update_centers, mean_costs, n_samples_per_update, max_eigval_per_update = plotEvolutionaryOptimization(n_updates,directories[dd])
        
        all_updates[dd] = update_centers
        all_mean_costs[dd] = mean_costs
        all_n_samples_per_update[dd] = n_samples_per_update
        all_max_eigval_per_update[dd] = max_eigval_per_update

    mean_mean_costs = all_mean_costs.mean(0)
    std_mean_costs = all_mean_costs.std(0)
    
    mean_max_eigval_per_update = all_max_eigval_per_update.mean(0)
    std_max_eigval_per_update = all_max_eigval_per_update.std(0)

    std_updates = all_updates.std(0)
    if (sum(std_updates)>0.0001):
        print "WARNING: updates must be the same"
      
    ax1 = axs[0]
    #ax1 = fig.add_subplot(121)
    lines_lc      = ax1.plot(n_samples_per_update,mean_max_eigval_per_update,linewidth=2)
    lines_lc.extend(ax1.plot(n_samples_per_update,mean_max_eigval_per_update+std_max_eigval_per_update,linewidth=1))
    lines_lc.extend(ax1.plot(n_samples_per_update,mean_max_eigval_per_update-std_max_eigval_per_update,linewidth=1))
    plt.setp(lines_lc,color='green')
    
    y_limits = [0,1.1*max(mean_max_eigval_per_update)];
    ax1.set_xlabel('number of evaluations')
    ax1.set_ylabel('maximum eigval of covar matrix')
    ax1.set_title('Exploration magnitude')
    plotUpdateLines(n_samples_per_update,y_limits,ax1)
    ax1.set_ylim(y_limits)

    ax2 = axs[1]
    #ax2 = fig.add_subplot(122)
    lines_ec      = ax2.plot(update_centers,mean_mean_costs,linewidth=2)
    lines_ec.extend(ax2.plot(update_centers,mean_mean_costs+std_mean_costs,linewidth=1))
    lines_ec.extend(ax2.plot(update_centers,mean_mean_costs-std_mean_costs,linewidth=1))
    plt.setp(lines_ec,color='blue')
    
    y_limits = [0,1.1*max(mean_mean_costs)];
    ax2.set_xlabel('number of evaluations')
    ax2.set_ylabel('mean cost per update')
    ax2.set_title('Learning curve')
    plotUpdateLines(n_samples_per_update,y_limits,ax2)
    ax2.set_ylim(y_limits)
    
    return (lines_lc, lines_ec, all_mean_costs)

    
if __name__=='__main__':
    
    # See if input directory was passed
    if (len(sys.argv)<2):
        print '\nUsage: '+sys.argv[0]+' <directory> [directory2] [directory3] [etc]\n';
        sys.exit()
        
    directories = []
    for arg in sys.argv[1:]:
        directories.append(str(arg))
    
    n_updates = 0;
    dir_exists = True;
    while (dir_exists):
      n_updates+=1
      cur_directory = '%s/update%05d' % (directories[0], n_updates)
      dir_exists = os.path.isdir(cur_directory)
    
    n_updates-=1
    
    

    fig = plt.figure(1,figsize=(12, 4))
    
    if (len(directories)==1):
        rollouts_python_script = directories[0]+'/plotRollouts.py'
        if (os.path.isfile(rollouts_python_script)):
            lib_path = os.path.abspath(directories[0])
            sys.path.append(lib_path)
            from plotRollouts import plotRollouts
            axs = [ fig.add_subplot(143), fig.add_subplot(144), fig.add_subplot(142) , fig.add_subplot(141) ]
        else:
            axs = [ fig.add_subplot(132), fig.add_subplot(133), fig.add_subplot(131) ]
        plotEvolutionaryOptimization(n_updates,directories[0],axs)
        #anim = animation.FuncAnimation(fig, plotEvolutionaryOptimization, frames=n_updates, fargs=(directory,), interval=100)
    else:
        axs = [ fig.add_subplot(121),  fig.add_subplot(122) ]
        plotEvolutionaryOptimizations(n_updates,directories,axs)
 
    plt.show()

