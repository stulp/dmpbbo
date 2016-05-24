## \file demoEvolutionaryOptimizationTask.py
## \author Freek Stulp
## \brief  Visualizes results of demoEvolutionaryOptimizationTask.cpp
## 
## \ingroup Demos
## \ingroup BBO

import matplotlib.pyplot as plt
import numpy
import subprocess

# Add relative path if PYTHONPATH is not set
import os, sys
lib_path = os.path.abspath('../../../python/')
sys.path.append(lib_path)

from dmp_bbo.dmp_bbo_plotting import plotOptimizationRollouts

def plotRollout(cost_vars,ax):
    line_handles = ax.plot(cost_vars.T,linewidth=0.5)
    return line_handles

if __name__=='__main__':
    executable = "../../../bin/demoOptimizationTask"
    
    if (not os.path.isfile(executable)):
        print("")
        print("ERROR: Executable '"+executable+"' does not exist.")
        print("Please call 'make install' in the build directory first.")
        print("")
        sys.exit(-1);
    
    covar_updates = ["none","decay","adaptation"]

    figure_number = 1;
    for covar_update in covar_updates:
      # Call the executable with the directory to which results should be written
      directory = "/tmp/demoOptimizationTask/"+covar_update
      command = executable+" "+directory+" "+covar_update
      print(command)
      subprocess.call(command, shell=True)
      
      fig = plt.figure(figure_number,figsize=(12, 4))
      figure_number += 1;
      plotOptimizationRollouts(directory,fig,plotRollout)
      fig.canvas.set_window_title("Optimization with covar_update="+covar_update) 

    plt.show()
    

