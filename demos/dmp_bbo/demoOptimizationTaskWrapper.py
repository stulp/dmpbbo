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


## \file demoEvolutionaryOptimizationTask.py
## \author Freek Stulp
## \brief  Visualizes results of demoEvolutionaryOptimizationTask.cpp
## 
## \ingroup Demos
## \ingroup BBO

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
    line_handles = ax.plot(cost_vars.T,linewidth=0.5)
    return line_handles

if __name__=='__main__':
    covar_updates = ["none","decay","adaptation"]

    figure_number = 1;
    for covar_update in covar_updates:
      # Call the executable with the directory to which results should be written
      executable = "../../bin/demoOptimizationTask"
      directory = "/tmp/demoOptimizationTask/"+covar_update
      arguments = directory+" "+covar_update
      executeBinary(executable,arguments,True)
      
      print("    Plotting")
      fig = plt.figure(figure_number,figsize=(12, 4))
      figure_number += 1;
      plotOptimizationRollouts(directory,fig,plotRollout)
      fig.canvas.set_window_title("Optimization with covar_update="+covar_update) 

    plt.show()
    

