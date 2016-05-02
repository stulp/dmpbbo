## \file demoOptimizationDmpParallel.py
## \author Freek Stulp
## \brief  Visualizes results of demoOptimizationDmpParallel.cpp
## 
## \ingroup Demos
## \ingroup DMP_BBO

import matplotlib.pyplot as plt
import numpy
import subprocess

# Add relative path if PYTHONPATH is not set
import os, sys
lib_path = os.path.abspath('../../../python/')
sys.path.append(lib_path)

from dmp_bbo.dmp_bbo_plotting import plotOptimizationRollouts

def plotRollout(cost_vars,ax):
    """Simple script to plot y of DMP trajectory"""
    n_dofs = (cost_vars.shape[1]-1)//4
    y = cost_vars[:,0:n_dofs]
    if n_dofs==1:
        line_handles = ax.plot(y,linewidth=0.5)
    else:
        line_handles = ax.plot(y[:,0],y[:,1],linewidth=0.5)
    return line_handles

if __name__=='__main__':
    executable = "../../../bin/demoOptimizationDmpParallel"
    
    if (not os.path.isfile(executable)):
        print ""
        print "ERROR: Executable '"+executable+"' does not exist."
        print "Please call 'make install' in the build directory first."
        print ""
        sys.exit(-1);
    
    # Call the executable with the directory to which results should be written
    directory = "/tmp/demoOptimizationDmpParallel"
    command = executable+" "+directory
    print command
    subprocess.call(command, shell=True)
      
    fig = plt.figure(1,figsize=(12, 4))
    plotOptimizationRollouts(directory,fig,plotRollout)
    plt.show()
    

