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


## \file demoTrainFunctionApproximators.py
## \author Freek Stulp
## \brief  Visualizes results of demoTrainFunctionApproximators.cpp
## 
## \ingroup Demos
## \ingroup FunctionApproximators

import matplotlib.pyplot as plt
import os, sys, subprocess

# Include scripts for plotting
lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from functionapproximators.functionapproximators_plotting import *

if __name__=='__main__':
    executable = "../../bin/demoTrainFunctionApproximators"
    
    if (not os.path.isfile(executable)):
        print("")
        print("ERROR: Executable '"+executable+"' does not exist.")
        print("Please call 'make install' in the build directory first.")
        print("")
        sys.exit(-1);
    
    # Call the executable with the directory to which results should be written
    directory = "/tmp/demoTrainFunctionApproximators"
    subprocess.call([executable, directory])
    
    # Plot the results in each directory
    function_approximator_names = ["WLS","LWR","LWPR","RRRFF","GMR","RBFN","GPR"]
        
    fig_number = 1;
    for name in function_approximator_names:
        fig = plt.figure(fig_number)

        directory_fa = directory +"/"+ name
        if (getDataDimFromDirectory(directory_fa)==1):
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')

        fig_number += 1;
    
        try:
            if (name=="WLS" or name=="LWR" or name=="LWPR" or name=="GMR"):
                plotLocallyWeightedLinesFromDirectory(directory_fa,ax)
            elif (name=="RBFN" or name=="GPR" or name=="RRRFF"):
                plotBasisFunctionsFromDirectory(directory_fa,ax)
            plotDataFromDirectory(directory_fa,ax)
            ax.set_ylim(-1.0,1.5)
        except IOError:
            print("WARNING: Could not find data for function approximator "+name)
        ax.set_title(name)
    
    ax.legend(['f(x)','+std','-std','residuals'])
    
    plt.show()
    
    #fig.savefig("lwr.svg")

