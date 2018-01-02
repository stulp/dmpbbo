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
import os, sys
import matplotlib.pyplot as plt
from pylab import mean

from matplotlib.patches import Ellipse

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from bbo.distribution_gaussian import DistributionGaussian

def setColor(handle,i_update,n_updates):
    if (i_update==0):
        plt.setp(handle,color='r',linewidth=2)
    else:
        color_val = (1.0*i_update/n_updates)
        cur_color = [0.0-0.0*color_val, 0.0+1.0*color_val, 0.0-0.0*color_val]
        plt.setp(handle,color=cur_color)

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


def plotLearningCurve(learning_curve,ax,costs_all=[],cost_labels=[]):

    # Plot (sum of) costs of all individual samples 
    if (len(costs_all)>0):
        # costs_all may also contain individual cost components. Only take
        # first one, because it represents the sum of the individual comps.
        costs_all = np.atleast_2d(costs_all)[:,0]
        ax.plot(costs_all,'.',color='gray')
        
    # Plot costs at evaluations 
    learning_curve = np.array(learning_curve)
    # Sum of cost components
    samples_eval = learning_curve[:,0]
    costs_eval = learning_curve[:,1:]
    lines = ax.plot(samples_eval,costs_eval[:,0],'-',color='black',linewidth=2)
    # Individual cost components
    if costs_eval.shape[1]>1:
        lines.extend(ax.plot(samples_eval,costs_eval[:,1:],'-',linewidth=1))
    
    # Annotation
    ax.set_xlabel('number of evaluations')
    ax.set_ylabel('cost')
    ax.set_title('Learning curve')
    
    if len(cost_labels)>0:
        cost_labels.insert(0,'total cost')
        plt.legend(lines, cost_labels)

    y_limits = [0,1.2*np.max(costs_eval)]
    ax.set_ylim(y_limits)
    return lines
    
    
def plotExplorationCurve(exploration_curve,ax):    
    exploration_curve = np.array(exploration_curve)
    line = ax.plot(exploration_curve[:,0],exploration_curve[:,1],'-',color='green',linewidth=2)
    ax.set_xlabel('number of evaluations')
    ax.set_ylabel('sqrt of max. eigval of covar')
    ax.set_title('Exploration magnitude')
    return line

def plotCurveDeprecated(curve,axs,costs_all=[]):
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

def saveLearningCurve(directory,learning_curve):
    np.savetxt(directory+'/learning_curve.txt',learning_curve)
    
def loadLearningCurve(directory):
    return np.loadtxt(directory+'/learning_curve.txt')

def saveExplorationCurve(directory,exploration_curve):
    np.savetxt(directory+'/exploration_curve.txt',exploration_curve)
    
def loadExplorationCurve(directory):
    return np.loadtxt(directory+'/exploration_curve.txt')
    
    
def saveUpdate(directory,i_update,distribution,cost_eval,samples,costs,weights,distribution_new):

    if not os.path.exists(directory):
        os.makedirs(directory)
        
    cur_dir = '%s/update%05d' % (directory, i_update)
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)
    
    np.savetxt(cur_dir+"/distribution_mean.txt",distribution.mean)
    np.savetxt(cur_dir+"/distribution_covar.txt",distribution.covar)
    np.savetxt(cur_dir+"/distribution_new_mean.txt",distribution_new.mean)
    np.savetxt(cur_dir+"/distribution_new_covar.txt",distribution_new.covar)
          
    if cost_eval is not None:
        np.savetxt(cur_dir+'/cost_eval.txt',np.atleast_1d(cost_eval))
    if samples is not None:
        np.savetxt(cur_dir+'/samples.txt',samples)
    if costs is not None:
        np.savetxt(cur_dir+'/costs.txt',costs)
    if weights is not None:
        np.savetxt(cur_dir+'/weights.txt',weights)
        
def plotLearningCurves(all_eval_at_samples,all_costs_eval,ax):
    
    # Check if all n_samples_per_update are the same. Otherwise averaging doesn't make sense.
    std_samples = all_eval_at_samples.std(0)
    if (sum(std_samples)>0.0001):
        print("WARNING: updates must be the same")
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
        print("WARNING: updates must be the same")
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
    
def plotUpdate(distribution,cost_eval,samples,costs,weights,distribution_new,ax,highlight=False,plot_samples=False):

    if samples is None:
        plot_samples = False
    
    n_dims = len(distribution.mean)
    if (n_dims==1):
        print("Sorry, only know how to plot for n_dims==2, but you provided n_dims==1")
        return
        
    # ZZZ Take into consideration block_covar_sizes to plot sub blocks in 
    # different subplots
    
    if (n_dims>=2):
        #print("Sorry, only know how to plot for n_dims==2, throwing away excess dimensions")
        distr_mean = distribution.mean[0:2]
        distr_covar = distribution.covar[0:2,0:2]
        distr_new_mean  = distribution_new.mean[0:2]
        distr_new_covar = distribution_new.covar[0:2,0:2]
        if samples is not None:
            samples = samples[:,0:2]
                    
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
        patch = plot_error_ellipse(distr_mean,distr_covar,ax)
        patch_new = plot_error_ellipse(distr_new_mean,distr_new_covar,ax)
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
