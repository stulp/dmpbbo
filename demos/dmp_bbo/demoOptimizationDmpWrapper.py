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


## \file demoOptimizationDmp.py
## \author Freek Stulp
## \brief  Visualizes results of demoOptimizationDmp.cpp
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
from dmp_bbo.tasks.TaskViapoint import TaskViapoint
from dmp_bbo.dmp_bbo_plotting import plotOptimizationRollouts
from dmp_bbo.dmp_bbo_plotting import plotOptimizationRolloutsTask

if __name__=='__main__':
    
    for n_dims in [1,2]:
        
        # Initialize a viapoint task and save it to file
        viapoint = 2*np.ones(n_dims)
        viapoint[0] = 2.5
        viapoint_time = 0.5
        viapoint_radius = 0.0
        if n_dims==2:
            # Do not pass through viapoint at a specific time, but rather pass
            # through it at any time.
            viapoint_time = None
        task = TaskViapoint(viapoint,viapoint_time, viapoint_radius)
        directory = "/tmp/demoOptimizationDmp/"+str(n_dims)+"D/"
        task.saveToFile(directory,"viapoint_task.txt")
        
        # Call the executable with the directory to which results should be written
        print("Call executable to run optimization with "+str(n_dims)+"D viapoint.")
        executable = "../../bin/demoOptimizationDmp"
        executeBinary(executable, str(n_dims)+" "+directory)
        
        print("  Plotting")
        fig = plt.figure(2*n_dims+0,figsize=(12, 4))
        plotOptimizationRolloutsTask(directory,fig,task)

    print("Showing")
    plt.show()
    

