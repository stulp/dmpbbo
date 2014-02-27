import sys
import numpy                                                                    
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import os
import matplotlib.pyplot as pl
import time
#from matplotlib import animation

# Include scripts for plotting
lib_path = os.path.abspath('../../bbo/plotting')
sys.path.append(lib_path)

from plotUpdateSummaryParallel import plotUpdateSummaryParallelFromDirectory
from plotEvolutionaryOptimization import *

def plotEvolutionaryOptimizationParallel(directory,axs,plot_all_rollouts=False):

    try:
        n_parallel = np.loadtxt(directory+"/n_parallel.txt")
    except IOError:
        n_parallel = 1
        
    #################################
    # Load and plot learning curve
    (update_at_samples, costs_all, eval_at_samples, costs_eval) = loadLearningCurve(directory)
    ax = (None if axs==None else axs[1])
    plotLearningCurve(eval_at_samples,costs_eval,ax,costs_all)
    #y_limits = [0,1.2*max(learning_curve)];
    plotUpdateLines(update_at_samples,ax)
    
    #################################
    # Load and plot exploration curve
    dim_dir = "";
    ax = (None if axs==None else axs[0])
    for i_parallel in range(n_parallel):
        (covar_at_samples, sqrt_max_eigvals) = loadExplorationCurve(directory,i_parallel)
        plotExplorationCurve(covar_at_samples,sqrt_max_eigvals,ax)
    plotUpdateLines(update_at_samples,ax)


    n_updates = loadNumberOfUpdates(directory)
    ax = (None if axs==None else axs[2])
    if (ax!=None):
        #################################
        # Visualize the update in parameter space 
        for update in range(n_updates):
            cur_directory = '%s/update%05d' % (directory, update+1)
            plotUpdateSummaryParallelFromDirectory(cur_directory,ax,False)
        ax.set_title('Search space')
          
    if ( (axs!=None) and (len(axs)>3) ):
      ax = axs[3];
      rollouts_python_script = directory+'/plotRollouts.py'
      print rollouts_python_script
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

def plotEvolutionaryOptimizationsParallel(directories,axs):
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
    


    try:
        n_parallel = np.loadtxt(directory+"/n_parallel.txt")
    except IOError:
        n_parallel = 1
        
    dim_dir = "";
    ax = (None if axs==None else axs[0])
    lines_ec = []
    for i_parallel in range(n_parallel):
        if (n_parallel>1):
            dim_dir = "dim%02d" % i_parallel
            
        all_sqrt_max_eigvals = np.empty((n_dirs,n_updates+1), dtype=float)
        all_covar_at_samples = np.empty((n_dirs,n_updates+1), dtype=float)
        for dd in range(len(directories)):
            (covar_at_samples, sqrt_max_eigvals) = loadExplorationCurve(directories[dd],dim_dir)
            all_sqrt_max_eigvals[dd] = sqrt_max_eigvals[:n_updates+1]
            all_covar_at_samples[dd] = covar_at_samples[:n_updates+1]
            
        ax = axs[0]
        more_lines_ec = plotExplorationCurves(all_covar_at_samples,all_sqrt_max_eigvals,ax)
        lines_ec.extend(more_lines_ec)
        
    plotUpdateLines(update_at_samples,ax)
    
    return (lines_lc, lines_ec)

    
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
        plotEvolutionaryOptimizationParallel(directories[0],axs)
        #anim = animation.FuncAnimation(fig, plotEvolutionaryOptimization, frames=n_updates, fargs=(directory,), interval=100)
    else:
        axs = [ fig.add_subplot(121),  fig.add_subplot(122) ]
        plotEvolutionaryOptimizationsParallel(directories,axs)
 
    plt.show()

