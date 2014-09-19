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
from plotData import getDataDimFromDirectory
from plotLocallyWeightedLines import plotLocallyWeightedLinesFromDirectory
from plotBasisFunctions import plotBasisFunctionsFromDirectory

if __name__=='__main__':
    executable = "../../../bin_test/testToModelParametersUnified"
    
    if (not os.path.isfile(executable)):
        print ""
        print "ERROR: Executable '"+executable+"' does not exist."
        print "Please call 'make install' in the build directory first."
        print ""
        sys.exit(-1);
    
    function_approximator_names = ["RBFN","LWR","GPR","GMR"]
    
    # Call the executable with the directory to which results should be written
    directory = "/tmp/testToModelParametersUnified"
    command = [executable,  directory] + function_approximator_names;
    subprocess.call(command)
    
    # Plot the results in each directory
    
    fig_number = 1;
    for name in function_approximator_names:
        fig = plt.figure(fig_number)
        fig_number += 1;
        
        subplot_number = 1;
        for uni_name in ['','Unified']:
            directory_fa = directory +"/"+ name + "1D" + uni_name
            print(directory_fa)
            if (getDataDimFromDirectory(directory_fa)==1):
                ax = fig.add_subplot(120+subplot_number)
            else:
                ax = fig.add_subplot(120+subplot_number, projection='3d')
            subplot_number = subplot_number+1
    
            try:
                plotDataFromDirectory(directory_fa,ax)
                if (uni_name=="Unified" or name=="WLS" or name=="LWR" or name=="LWPR" or name=="GPR"):
                    plotLocallyWeightedLinesFromDirectory(directory_fa,ax)
                elif (name=="RBFN"):
                    plot_normalized=False
                    plotBasisFunctionsFromDirectory(directory_fa,ax,plot_normalized)
                ax.set_ylim(-1.0,1.5)
            except IOError:
                print "WARNING: Could not find data for function approximator "+name
            ax.set_title(name+" "+uni_name)
        
        #ax.legend(['f(x)','+std','-std','residuals'])
    
    plt.show()
    
    #fig.savefig("lwr.svg")

