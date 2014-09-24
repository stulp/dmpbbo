from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy                                                                  
import matplotlib.pyplot as plt                                               
import os, sys, subprocess

lib_path = os.path.abspath('../plotting')
sys.path.append(lib_path)

from plotData import getDataDimFromDirectory
from plotData import plotDataFromDirectory
from plotLocallyWeightedLines import plotLocallyWeightedLinesFromDirectory


def plotFunctionApproximatorTrainingFromDirectory(directory,ax,ax2=None):
    """Load data related to function approximator training from a directory and plot it."""
    if ax2 != None:
        plotLocallyWeightedLinesFromDirectory(directory,ax2)
    else:
        plotLocallyWeightedLinesFromDirectory(directory,ax)
        
    plotDataFromDirectory(directory,ax)
    
        
    

if __name__=='__main__':
    """Run some training sessions and plot results."""

    executable = "../../../bin_test/testFunctionApproximatorTraining"
    
    if (not os.path.isfile(executable)):
        print ""
        print "ERROR: Executable '"+executable+"' does not exist."
        print "Please call 'make install' in the build directory first."
        print ""
        sys.exit(-1);
    
    fig_number = 1;     
    directory = "/tmp/testFunctionApproximatorTraining/"
    
    fa_names = ["RBFN","GPR","IRFRLS","LWR", "LWPR", "GMR"] 
    for fa_name in fa_names:
      
        # Call the executable with the directory to which results should be written
        command = executable+" "+directory+" "+fa_name
        #print command
        subprocess.call(command, shell=True)
    
    for fa_name in fa_names:
        print("Plotting "+fa_name+" results")
        fig = plt.figure(fig_number,figsize=(15,5))
        fig_number = fig_number+1
        for dim in [1, 2]:
            
            cur_directory = directory+fa_name+"_"+str(dim)+"D";
            if (getDataDimFromDirectory(cur_directory)==1):
                ax = fig.add_subplot(1, 3, 1)
                ax2 = None
            else:
                ax = fig.add_subplot(1, 3, 2, projection='3d')
                ax2 = fig.add_subplot(1, 3, 3, projection='3d')
            plotFunctionApproximatorTrainingFromDirectory(cur_directory,ax,ax2)
            ax.set_title(fa_name+" ("+str(dim)+"D data)")
            if ax2 != None:
                ax2.set_title(fa_name+" ("+str(dim)+"D basis functions)")
              
          
    plt.show()


