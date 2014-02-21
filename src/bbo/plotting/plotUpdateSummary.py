import sys
import numpy                                                                    
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import os
import matplotlib.pyplot as pl
from matplotlib.patches import Ellipse
import time
#from matplotlib import animation

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
    
    
def plotUpdateSummary(distribution_mean,distribution_covar,samples,costs,weights,distribution_new_mean,distribution_new_covar,ax,highlight=True,plot_samples=False):
    n_dims = len(distribution_mean);
    if (n_dims==1):
        print "Sorry, only know how to plot for n_dims==2, but you provided n_dims==1"
        return
    if (n_dims>2):
        #print "Sorry, only know how to plot for n_dims==2, throwing away excess dimensions"
        distribution_mean = distribution_mean[0:2]
        distribution_covar = distribution_covar[0:2,0:2]
        distribution_new_mean  = distribution_new_mean[0:2]
        distribution_new_covar = distribution_new_covar[0:2,0:2]
        samples = samples[:,0:2]
                
    if (plot_samples):
        max_marker_size = 80;
        for ii in range(len(weights)):
            cur_marker_size = max_marker_size*weights[ii]
            sample_handle = ax.plot(samples[ii,0],samples[ii,1],'o',color='green')
            plt.setp(sample_handle,markersize=cur_marker_size,markerfacecolor=(0.5,0.8,0.5),markeredgecolor='none')
      
            ax.plot(samples[:,0],samples[:,1],'.',color='black')
        ax.plot((distribution_mean[0],distribution_new_mean[0]),(distribution_mean[1],distribution_new_mean[1]),'-',color='blue')
            
    mean_handle = ax.plot(distribution_mean[0],distribution_mean[1],'o',label='old')
    mean_handle_new = ax.plot(distribution_new_mean[0],distribution_new_mean[1],'o',label='new')
    patch = plot_error_ellipse(distribution_mean[0:2],distribution_covar[0:2,0:2],ax)
    patch_new = plot_error_ellipse(distribution_new_mean[0:2],distribution_new_covar[0:2,0:2],ax)
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
    ax.set_aspect('equal')

    plt.rcParams['text.usetex']=True
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    return mean_handle,mean_handle_new,patch,patch_new
    
def plotUpdateSummaryFromDirectory(directory,ax,highlight=True,plot_samples=False):
    # Read data
    distribution_mean = np.loadtxt(directory+"/distribution_mean.txt")
    distribution_covar = np.loadtxt(directory+"/distribution_covar.txt")
    samples = np.loadtxt(directory+"/samples.txt")
    costs = np.loadtxt(directory+"/costs.txt")
    weights = np.loadtxt(directory+"/weights.txt")
    distribution_new_mean = np.loadtxt(directory+"/distribution_new_mean.txt")
    distribution_new_covar = np.loadtxt(directory+"/distribution_new_covar.txt")
    plotUpdateSummary(distribution_mean,distribution_covar,samples,costs,weights,distribution_new_mean,distribution_new_covar,ax,highlight,plot_samples)

if __name__=='__main__':
    
    # See if input directory was passed
    if (len(sys.argv)==2):
      directory = str(sys.argv[1])
    else:
      print '\nUsage: '+sys.argv[0]+' <directory>\n';
      sys.exit()
    
    fig = plt.figure()
    ax = fig.gca()
    highlight = True
    plot_samples = True
    plotUpdateSummaryFromDirectory(directory,ax,highlight,plot_samples)
    
  
    plt.show()

