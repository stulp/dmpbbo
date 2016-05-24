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

if __name__=='__main__':
    executable = "../../../bin_test/testBasisFunctionsLWR"
    
    if (not os.path.isfile(executable)):
        print("")
        print("ERROR: Executable '"+executable+"' does not exist.")
        print("Please call 'make install' in the build directory first.")
        print("")
        sys.exit(-1);
    
    # Call the executable with the directory to which results should be written
    directory = "/tmp/testBasisFunctionsLWR"
    subprocess.call([executable, directory])
    
    # Plot the results in each directory
    fig = plt.figure()
    subplot_number = 1;
    for sym_label in ["symmetric","Asymmetric"]:
        ax = fig.add_subplot(1,2,subplot_number)
        subplot_number += 1;
        directory_fa = directory +"/1D_" + sym_label
        plotDataFromDirectory(directory_fa,ax)
        plotLocallyWeightedLinesFromDirectory(directory_fa,ax)
        #ax.legend(['targets','predictions'])
        ax.set_title(sym_label)
    plt.show()
    

