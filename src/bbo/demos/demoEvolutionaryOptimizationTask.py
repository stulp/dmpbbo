## \file demoExponentialSystem.py
## \author Freek Stulp
## \brief  Visualizes results of demoExponentialSystem.cpp
## 
## \ingroup Demos
## \ingroup DynamicalSystems

import matplotlib.pyplot as plt
import numpy
import os, sys, subprocess

# Include scripts for plotting
lib_path = os.path.abspath('../plotting')
sys.path.append(lib_path)

from plotEvolutionaryOptimization import plotEvolutionaryOptimization

if __name__=='__main__':
    executable = "../../../bin/demoEvolutionaryOptimizationTask"
    
    if (not os.path.isfile(executable)):
        print ""
        print "ERROR: Executable '"+executable+"' does not exist."
        print "Please call 'make install' in the build directory first."
        print ""
        sys.exit(-1);
    
    covar_updates = ["none","decay","adaptation"]

    figure_number = 1;
    for covar_update in covar_updates:
      # Call the executable with the directory to which results should be written
      directory = "/tmp/demoEvolutionaryOptimizationTask/"+covar_update
      command = executable+" "+directory+" "+covar_update
      print command
      subprocess.call(command, shell=True)
      
      n_updates = 40 
      fig = plt.figure(figure_number,figsize=(12, 4))
      figure_number += 1;
      axs = [ fig.add_subplot(143), fig.add_subplot(144), fig.add_subplot(142) , fig.add_subplot(141) ]
      plotEvolutionaryOptimization(n_updates,directory,axs)
      fig.canvas.set_window_title("Optimization with covar_update="+covar_update) 

    plt.show()
    

