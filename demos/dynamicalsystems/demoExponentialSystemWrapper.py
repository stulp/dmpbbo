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


## \file demoExponentialSystem.py
## \author Freek Stulp
## \brief  Visualizes results of demoExponentialSystem.cpp
## 
## \ingroup Demos
## \ingroup DynamicalSystems

import matplotlib.pyplot as plt
import numpy
import os, sys

lib_path = os.path.abspath('../')
sys.path.append(lib_path)
from executeBinary import executeBinary

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)
from dynamicalsystems.dynamicalsystems_plotting import * 

if __name__=='__main__':
    
    # Call the executable with the directory to which results should be written
    executable = "./demoExponentialSystem"
    directory = "./demoExponentialSystemDataTmp"
    executeBinary(executable, directory)
    
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
    

