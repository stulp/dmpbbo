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


## \file demoOptimizationDmpParallel.py
## \author Freek Stulp
## \brief  Visualizes results of demoOptimizationDmpParallel.cpp
## 
## \ingroup Demos
## \ingroup DMP_BBO

import matplotlib.pyplot as plt
import numpy as np
import os, sys

lib_path = os.path.abspath('../')
sys.path.append(lib_path)
from executeBinary import executeBinary

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)
from dmp_bbo.dmp_bbo_plotting import plotOptimizationRollouts
from dmp_bbo.dmp_bbo_plotting import plotOptimizationRolloutsTask
from dmp_bbo.tasks.TaskViapoint import TaskViapoint

if __name__=='__main__':
    
    # Initialize a viapoint task and save it to file
    n_dims = 2
    viapoint = np.linspace(1.5,2,n_dims)
    viapoint_time = 0.2
    viapoint_radius = 0.0
    task = TaskViapoint(viapoint,viapoint_time, viapoint_radius)
    directory = "/tmp/demoOptimizationDmpParallel/"+str(n_dims)+"D/"
    task.saveToFile(directory,"viapoint_task.txt")
    
    # Call the executable with the directory to which results should be written
    executable = "../../bin/demoOptimizationDmpParallel"
    executeBinary(executable, directory)
      
    fig = plt.figure(1,figsize=(12, 4))
    plotOptimizationRolloutsTask(directory,fig,task)
    plt.show()
    

