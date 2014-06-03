## \file demoTrainFunctionApproximators.py
## \author Freek Stulp
## \brief  Visualizes results of demoTrainFunctionApproximators.cpp
## 
## \ingroup Demos
## \ingroup FunctionApproximators

import matplotlib.pyplot as plt
import os, sys, subprocess

# Include scripts for plotting
lib_path = os.path.abspath('../plotting')
sys.path.append(lib_path)
from plotData import plotDataFromDirectory
from plotLocallyWeightedLines import plotLocallyWeightedLinesFromDirectory
from plotBasisFunctions import plotBasisFunctionsFromDirectory

if __name__=='__main__':
    executable = "../../../bin/demoTrainFunctionApproximators"
    
    if (not os.path.isfile(executable)):
        print ""
        print "ERROR: Executable '"+executable+"' does not exist."
        print "Please call 'make install' in the build directory first."
        print ""
        sys.exit(-1);
    
    # Call the executable with the directory to which results should be written
    directory = "/tmp/demoTrainFunctionApproximators"
    subprocess.call([executable, directory])
    
    # Plot the results in each directory
    function_approximator_names = ["LWR","LWPR","IRFRLS","GMR","RBFN"]
    fig = plt.figure()
    subplot_number = 1;
    for name in function_approximator_names:
    
    
        ax = fig.add_subplot(1,len(function_approximator_names),subplot_number)
        subplot_number += 1;
    
        directory_fa = directory +"/"+ name
        try:
            plotDataFromDirectory(directory_fa,ax)
            if (name=="LWR" or name=="LWPR"):
                plotLocallyWeightedLinesFromDirectory(directory_fa,ax)
            elif (name=="RBFN"):
                plot_normalized=False
                plotBasisFunctionsFromDirectory(directory_fa,ax,plot_normalized)
            ax.set_ylim(-1.0,1.5)
        except IOError:
            print "WARNING: Could not find data for function approximator "+name
        ax.set_title(name)
    
    ax.legend(['targets','predictions'])
    plt.show()
    
    #fig.savefig("lwr.svg")

