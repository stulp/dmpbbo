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


## \file demoEvolutionaryOptimization.py
## \author Freek Stulp
## \brief  Visualizes results of demoEvolutionaryOptimization.cpp
## 
## \ingroup Demos
## \ingroup BBO

import matplotlib.pyplot as plt
import numpy as np
import subprocess

# Add relative path if PYTHONPATH is not set
import os, sys
lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from bbo.bbo_plotting import plotLearningCurve, plotExplorationCurve


if __name__=='__main__':
    executable = "../../bin/demoOptimization"
    
    if (not os.path.isfile(executable)):
        print("")
        print("ERROR: Executable '"+executable+"' does not exist.")
        print("Please call 'make install' in the build directory first.")
        print("")
        sys.exit(-1);
    
    covar_updates = ["none","decay","adaptation"]

    figure_number = 1;
    for covar_update in covar_updates:
        # Call the executable with the directory to which results should be written
        directory = "/tmp/demoOptimization/"+covar_update
        command = executable+" "+directory+" "+covar_update
        print(command)
        subprocess.call(command, shell=True)
      
        fig = plt.figure(figure_number)
        figure_number += 1;
        exploration_curve = np.loadtxt(directory+'/exploration_curve.txt')
        plotExplorationCurve(exploration_curve,fig.add_subplot(121))
        learning_curve = np.loadtxt(directory+'/learning_curve.txt')
        plotLearningCurve(learning_curve,fig.add_subplot(122))
        fig.canvas.set_window_title("Optimization with covar_update="+covar_update) 
      

    plt.show()
    

