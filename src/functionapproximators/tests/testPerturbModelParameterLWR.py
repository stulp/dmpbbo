# This file is part of DmpBbo, a set of libraries and programs for the 
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
# 
# DmpBbo is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# DmpBbo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.


from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy                                                                  
import matplotlib.pyplot as plt                                               
import os, sys, subprocess

# Include scripts for plotting
lib_path = os.path.abspath('../../../python')
sys.path.append(lib_path)

from functionapproximators.functionapproximators_plotting import *


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

    directory = "../../../build_dir_debug/src/functionapproximators/tests/"
    executable = directory+"testPerturbModelParameterLWR"
    
    if (not os.path.isfile(executable)):
        print("")
        print("ERROR: Executable '"+executable+"' does not exist.")
        print("Please call 'make install' in the build directory first.")
        print("")
        sys.exit(-1);
    
    fig_number = 1;     
    directory = "/tmp/testPerturbModelParameterLWR/"
    
  
    # Call the executable with the directory to which results should be written
    command = executable+" "+directory
    #print(command)
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


