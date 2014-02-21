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

from plotDynamicalSystem import plotDynamicalSystem
from plotDynamicalSystemComparison import plotDynamicalSystemComparison

if __name__=='__main__':
    executable = "../../../bin/demoExponentialSystem"
    
    if (not os.path.isfile(executable)):
        print ""
        print "ERROR: Executable '"+executable+"' does not exist."
        print "Please call 'make install' in the build directory first."
        print ""
        sys.exit(-1);
    
    # Call the executable with the directory to which results should be written
    directory = "/tmp/demoExponentialSystem"
    subprocess.call([executable, directory])
    
    fig = plt.figure(1)
    data_ana = numpy.loadtxt(directory+"/analytical.txt")
    plotDynamicalSystem(data_ana,[fig.add_subplot(1,2,1), fig.add_subplot(1,2,2)])
    plt.title('analytical')

    fig = plt.figure(2)
    data_num = numpy.loadtxt(directory+"/numerical.txt")
    plotDynamicalSystem(data_num,[fig.add_subplot(1,2,1), fig.add_subplot(1,2,2)])
    plt.title('numerical')

    fig = plt.figure(3)
    axs      =  [fig.add_subplot(2,2,1), fig.add_subplot(2,2,2)]
    axs_diff =  [fig.add_subplot(2,2,3), fig.add_subplot(2,2,4)]
    plotDynamicalSystemComparison(data_ana,data_num,'analytical','numerical',axs,axs_diff)
    axs[1].legend()

    plt.show()
    

