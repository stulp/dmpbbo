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
    

