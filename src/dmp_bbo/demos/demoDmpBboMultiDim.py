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
lib_path = os.path.abspath('../../dmp_bbo/plotting')
sys.path.append(lib_path)

from plotEvolutionaryOptimizationParallel import plotEvolutionaryOptimizationParallel

if __name__=='__main__':
    executable = "../../../bin/demoDmpBboMultiDim"
    
    if (not os.path.isfile(executable)):
        print ""
        print "ERROR: Executable '"+executable+"' does not exist."
        print "Please call 'make install' in the build directory first."
        print ""
        sys.exit(-1);
    
    # Call the executable with the directory to which results should be written
    directory = "/tmp/demoDmpBboMultiDim/"
    command = executable+" "+directory
    print command
    subprocess.call(command, shell=True)
      
    fig = plt.figure(1,figsize=(12, 4))
    axs = [ fig.add_subplot(143), fig.add_subplot(144), fig.add_subplot(142) , fig.add_subplot(141) ]
    plotEvolutionaryOptimizationParallel(directory,axs)
    plt.show()
    

