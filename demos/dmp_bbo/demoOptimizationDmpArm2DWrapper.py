# This file is part of DmpBbo, a set of libraries and programs for the 
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2018 Freek Stulp
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

def plotRollout(cost_vars,ax):
    """Simple script to plot y of DMP trajectory"""
    n_cost_vars = cost_vars.shape[1]
    # cost_vars is assumed to have following structure
    # time  joint angles (e.g. n_dofs = 3)     forcing term  link positions (e.g. 3+1) 
    # ____  __________________________________  __________  __________________________    
    # t     | a a a | ad ad ad | add add add |  f f f       | x y | x y | x y | x y  |     
    #
    # 1     + 3*n_dofs                        + n_dofs +       2*(n_dofs+1))
    # n_cost_vars == 1 + 3*n_dofs + n_dofs + 2*(n_dofs+1);
    #  ergo...
    # n_cost_vars  == 1+ 3*n_dofs + n_dofs + 2*n_dofs+2;
    # n_cost_vars  == 3 + 6*n_dofs;                            
    # n_cost_vars-3  == 6*n_dofs;                            
    # (n_cost_vars-3)/6  == n_dofs;                            
    n_dofs = (n_cost_vars-3)//6
    t = cost_vars[:,0]
    x_endeff = cost_vars[:,1+6*n_dofs+0]
    y_endeff = cost_vars[:,1+6*n_dofs+1]
    
    #line_handles = ax.plot(x_endeff,y_endeff,linewidth=0.5)
    #line_handles = ax.plot(t,cost_vars[:,1:1+n_dofs],linewidth=0.5)
    line_handles = ax.plot(x_endeff,y_endeff,linewidth=0.5)
    ax.plot(0.5, 0.5, 'or')
    ax.axis('equal')
    return line_handles

if __name__=='__main__':
    
    for n_dofs in [3]:
        
        # Initialize a viapoint task and save it to file
        viapoint = 0.5*np.ones(2) # Always 2D!
        task = TaskViapoint(viapoint)
        directory = "/tmp/demoOptimizationDmpArm2D/"+str(n_dofs)+"D/"
        task.saveToFile(directory,"viapoint_task.txt")
        
        # Call the executable with the directory to which results should be written
        print("Call executable to run optimization with "+str(n_dofs)+"D viapoint.")
        executable = "./demoOptimizationDmpArm2D"
        print(executeBinary(executable, str(n_dofs)+" "+directory))
        
        print("  Plotting")
        fig = plt.figure(n_dofs,figsize=(12, 4))
        plotOptimizationRollouts(directory,fig,plotRollout)
        #plotOptimizationRolloutsTask(directory,fig,task)

    print("Showing")
    plt.show()
    

