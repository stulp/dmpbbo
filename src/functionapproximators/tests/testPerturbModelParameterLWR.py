from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy                                                                  
import matplotlib.pyplot as plt                                               
import os, sys, subprocess

lib_path = os.path.abspath('../plotting')
sys.path.append(lib_path)

from plotData import getDataDimFromDirectory
from plotData import plotDataFromDirectory
from plotLocallyWeightedLines import plotLocallyWeightedLinesFromDirectory


def plotFunctionApproximatorTrainingFromDirectory(directory,ax):
    """Load data related to function approximator training from a directory and plot it."""
    plotDataFromDirectory(directory,ax)
    
    data_read_successfully = True
    cur_directory_number=0
    while (data_read_successfully):
      cur_dir = directory+'/perturbation'+str(cur_directory_number)+'/'
      data_read_successfully = plotLocallyWeightedLinesFromDirectory(cur_dir,ax)
      cur_directory_number+=1

if __name__=='__main__':
    """Pass a directory argument, read inputs, targets and predictions from that directory, and plot."""

    executable = "../../../bin_test/testPerturbModelParameterLWR"
    
    if (not os.path.isfile(executable)):
        print ""
        print "ERROR: Executable '"+executable+"' does not exist."
        print "Please call 'make install' in the build directory first."
        print ""
        sys.exit(-1);
    
    fig_number = 1;     
    directory = "/tmp/testPerturbModelParameterLWR/"
    
  
    # Call the executable with the directory to which results should be written
    command = executable+" "+directory
    #print command
    subprocess.call(command, shell=True)
  
    for dim in [1, 2]:
        fig = plt.figure(dim)
        cur_directory = directory+"/"+str(dim)+"D"
        if (getDataDimFromDirectory(cur_directory)==1):
          ax = fig.gca()
        else:
          ax = Axes3D(fig)
      
        plotFunctionApproximatorTrainingFromDirectory(cur_directory,ax)
        
    plt.show()


