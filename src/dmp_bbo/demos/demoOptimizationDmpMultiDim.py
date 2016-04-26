## \file demoOptimizationDmpMultiDim.py
## \author Freek Stulp
## \brief  Visualizes results of demoOptimizationDmpMultiDim.cpp
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

import bbo.bbo_plotting

if __name__=='__main__':
    executable = "../../../bin/demoOptimizationDmpMultiDim"
    
    if (not os.path.isfile(executable)):
        print ""
        print "ERROR: Executable '"+executable+"' does not exist."
        print "Please call 'make install' in the build directory first."
        print ""
        sys.exit(-1);
    
    # Call the executable with the directory to which results should be written
    directory = "/tmp/demoOptimizationDmpMultiDim"
    command = executable+" "+directory
    print command
    subprocess.call(command, shell=True)
      
    fig = plt.figure(1,figsize=(12, 4))
    axs = [ fig.add_subplot(143), fig.add_subplot(144), fig.add_subplot(142) , fig.add_subplot(141) ]
    bbo.bbo_plotting.plotEvolutionaryOptimizationDir(directory,axs)
    plt.show()
    

