import numpy as np
import matplotlib.pyplot as plt
from pylab import mean

def plotUpdateLines(n_samples_per_update,ax,y_limits=[]):
    if (len(y_limits)==0):
        y_limits = ax.get_ylim()
    
    # Find good number of horizontal update lines to plot    
    updates = np.arange(0, len(n_samples_per_update))
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

    
def plotExplorationCurve(n_samples_per_update,exploration_curve,ax):
    line = ax.plot(n_samples_per_update,exploration_curve,'-',color='green',linewidth=2)
    ax.set_xlabel('number of evaluations')
    ax.set_ylabel('sqrt of max. eigval of covar')
    ax.set_title('Exploration magnitude')
    return line

def plotCurve(curve,axs,costs_all=[]):
    lines = []
    if curve.shape[1]>2 and len(axs)>=2: # Plot exploration too?
        ax_explo = axs[0]
        ax_cost = axs[1]
        
        line = plotExplorationCurve(curve[:,0],curve[:,2],ax_explo)
        plotUpdateLines(curve[:,0],ax_explo)
        lines.append(line)
        
    else:
        ax_cost = axs[0]
        
    # Plot costs of all individual samples 
    if (len(costs_all)>0):
        ax_cost.plot(costs_all,'.',color='gray')
    # Plot costs at evaluations 
    line = plotLearningCurve(curve[:,0],curve[:,1],ax_cost)
    plotUpdateLines(curve[:,0],ax_cost)
    
    lines.append(line)
    return lines


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

