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
import numpy
import os, sys

lib_path = os.path.abspath('../')
sys.path.append(lib_path)
from executeBinary import executeBinary

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)
from dmp_bbo.dmp_bbo_plotting import plotOptimizationRollouts

def plotRollout(cost_vars,ax):
    """Simple script to plot y of DMP trajectory"""
    n_dofs = (cost_vars.shape[1]-1)//4
    y = cost_vars[:,0:n_dofs]
    if n_dofs==1:
        line_handles = ax.plot(y,linewidth=0.5)
    else:
        line_handles = ax.plot(y[:,0],y[:,1],linewidth=0.5)
    return line_handles

if __name__=='__main__':
    # Call the executable with the directory to which results should be written
    directory = "/tmp/demoOptimizationDmpParallel"
    executable = "../../bin/demoOptimizationDmpParallel"
    executeBinary(executable, directory)
      
    fig = plt.figure(1,figsize=(12, 4))
    plotOptimizationRollouts(directory,fig,plotRollout)
    plt.show()
    

