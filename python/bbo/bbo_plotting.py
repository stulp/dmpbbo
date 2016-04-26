import numpy as np
import matplotlib.pyplot as plt
from pylab import mean

from matplotlib.patches import Ellipse

from bbo.distribution_gaussian import DistributionGaussian

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




# From https://github.com/dfm/dfmplot/blob/master/dfmplot/ellipse.py
def plot_error_ellipse(mu, cov, ax=None, **kwargs):
    """
Plot the error ellipse at a point given it's covariance matrix

Parameters
----------
mu : array (2,)
The center of the ellipse

cov : array (2,2)
The covariance matrix for the point

ax : matplotlib.Axes, optional
The axis to overplot on

**kwargs : dict
These keywords are passed to matplotlib.patches.Ellipse

"""
    # some sane defaults
    facecolor = kwargs.pop('facecolor', 'none')
    edgecolor = kwargs.pop('edgecolor', 'k')

    x, y = mu
    U,S,V = np.linalg.svd(cov)
    theta = np.degrees(np.arctan2(U[1,0], U[0,0]))
    ellipsePlot = Ellipse(xy=[x, y],
            width = 2*np.sqrt(S[0]),
            height= 2*np.sqrt(S[1]),
            angle=theta,
            facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    if ax is None:
        ax = pl.gca()
    lines = ax.add_patch(ellipsePlot)
    return lines
    
def plotUpdate(distributions,cost_eval,samples,costs,weights,distributions_new,ax,highlight=False,plot_samples=False):
    
    if isinstance(distributions,DistributionGaussian):
        # distributions should be a list of DistributionGaussian
        distributions = [distributions]
        
    if isinstance(distributions_new,DistributionGaussian):
        # distributions should be a list of DistributionGaussian
        distributions_new = [distributions_new]
    
    if samples==None:
        plot_samples = False
    
    n_parallel = len(distributions)
    n_dims = len(distributions[0].mean)
    if (n_dims==1):
        print "Sorry, only know how to plot for n_dims==2, but you provided n_dims==1"
        return
    for i_parallel in range(n_parallel):
        # ZZZ Perhaps plot in different subplots
        if (n_dims>=2):
            #print "Sorry, only know how to plot for n_dims==2, throwing away excess dimensions"
            distr_mean = distributions[i_parallel].mean[0:2]
            distr_covar = distributions[i_parallel].covar[0:2,0:2]
            distr_new_mean  = distributions_new[i_parallel].mean[0:2]
            distr_new_covar = distributions_new[i_parallel].covar[0:2,0:2]
            if samples!=None:
                samples = samples[i_parallel,0:2]
                    
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

    #plt.rcParams['text.usetex']=True
    #ax.set_xlabel(r'$\theta_1$')
    #ax.set_ylabel(r'$\theta_2$')
    ax.set_xlabel('dim 1 (of '+str(n_dims)+')')
    ax.set_ylabel('dim 2 (of '+str(n_dims)+')')
    ax.set_title('Search space')
    return mean_handle,mean_handle_new,patch,patch_new
