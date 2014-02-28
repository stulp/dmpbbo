import sys
import os
import numpy as np

# Include scripts for plotting
lib_path = os.path.abspath('../../bbo/plotting')
sys.path.append(lib_path)

from plotUpdateSummary import plotUpdateSummary

def plotUpdateSummaryParallelFromDirectory(directory,ax,highlight=True,plot_samples=False):
    # Read data
    costs = np.loadtxt(directory+"/costs.txt")
    weights = np.loadtxt(directory+"/weights.txt")
    n_parallel = np.loadtxt(directory+"/n_parallel.txt")
    for i_parallel in range(n_parallel):
        suffix = '_%02d' % i_parallel
        distribution_mean = np.loadtxt(directory+"/distribution_mean"+suffix+".txt")
        distribution_covar = np.loadtxt(directory+"/distribution_covar"+suffix+".txt")
        samples = np.loadtxt(directory+"/samples"+suffix+".txt")
        distribution_new_mean = np.loadtxt(directory+"/distribution_new_mean"+suffix+".txt")
        distribution_new_covar = np.loadtxt(directory+"/distribution_new_covar"+suffix+".txt")
        plotUpdateSummary(distribution_mean,distribution_covar,samples,costs,weights,distribution_new_mean,distribution_new_covar,ax,highlight,plot_samples)

