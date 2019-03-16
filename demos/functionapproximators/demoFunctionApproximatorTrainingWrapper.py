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
import os, sys

lib_path = os.path.abspath('../')
sys.path.append(lib_path)
from executeBinary import executeBinary

lib_path = os.path.abspath('../../python')
sys.path.append(lib_path)
from functionapproximators.functionapproximators_plotting import *


def plotFunctionApproximatorTrainingFromDirectory(directory,ax,ax2=None):
    """Load data related to function approximator training from a directory and plot it."""
    if ax2 != None:
        plotLocallyWeightedLinesFromDirectory(directory,ax2)
    else:
        plotLocallyWeightedLinesFromDirectory(directory,ax)
        
    plotDataFromDirectory(directory,ax)
    
        
    

if __name__=='__main__':
    """Run some training sessions and plot results."""

    fig_number = 1;     
    executable = "./demoFunctionApproximatorTraining"
    directory = "/tmp/demoFunctionApproximatorTraining/"
    
    fa_names = ["RBFN","GPR","RRRFF","LWR", "LWPR", "GMR"] 
    for fa_name in fa_names:
      
        # Call the executable with the directory to which results should be written
        arguments = directory+" "+fa_name
        executeBinary(executable, arguments)
    
    
    for fa_name in fa_names:
        cur_directory = directory+fa_name+"_1D";
        if not os.path.exists(cur_directory):
            break
        
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


