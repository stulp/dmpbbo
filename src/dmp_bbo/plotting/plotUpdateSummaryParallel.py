import sys
import os
import numpy as np
import warnings

# Include scripts for plotting
lib_path = os.path.abspath('../../bbo/plotting')
sys.path.append(lib_path)

from plotUpdateSummary import plotUpdateSummary

def plotUpdateSummaryParallelFromDirectory(directory,ax,highlight=True,plot_samples=False):
    # Read data
    costs = np.loadtxt(directory+"/costs.txt")
    # Supress warnings http://stackoverflow.com/questions/19167550/prevent-or-dismiss-empty-file-warning-in-loadtxt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weights = np.loadtxt(directory+"/weights.txt")
    if not weights:
        # Sometimes weights.txt will be empty. Fill it with 1 in this case. 
        weights = np.empty(len(costs)); 
        weights.fill(1) 
    
    n_parallel = np.loadtxt(directory+"/n_parallel.txt")
    for i_parallel in range(n_parallel):
        suffix = '_%02d' % i_parallel
        distribution_mean = np.loadtxt(directory+"/distribution_mean"+suffix+".txt")
        distribution_covar = np.loadtxt(directory+"/distribution_covar"+suffix+".txt")
        samples = np.loadtxt(directory+"/samples"+suffix+".txt")
        distribution_new_mean = np.loadtxt(directory+"/distribution_new_mean"+suffix+".txt")
        distribution_new_covar = np.loadtxt(directory+"/distribution_new_covar"+suffix+".txt")
        plotUpdateSummary(distribution_mean,distribution_covar,samples,costs,weights,distribution_new_mean,distribution_new_covar,ax,highlight,plot_samples)

