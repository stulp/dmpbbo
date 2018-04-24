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


## \file demoDmpChangeGoal.py
## \author Freek Stulp
## \brief  Visualizes results of demoDmpChangeGoal.cpp
## 
## \ingroup Demos
## \ingroup Dmps

import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys, subprocess

lib_path = os.path.abspath('../../src/dmp/plotting')
sys.path.append(lib_path)
lib_path = os.path.abspath('../../src/dynamicalsystems/plotting')
sys.path.append(lib_path)

from plotTrajectory import plotTrajectoryFromFile
from plotTrajectory import plotTrajectory
from plotDmp import plotDmp

executable = "../../bin/demoDmpChangeGoal"

if (not os.path.isfile(executable)):
    print("")
    print("ERROR: Executable '"+executable+"' does not exist.")
    print("Please call 'make install' in the build directory first.")
    print("")
    sys.exit(-1);

# Call the executable with the directory to which results should be written
directory = "/tmp/demoDmpChangeGoal"
subprocess.call([executable, directory])

print("Plotting")

for numerical_or_analytical in [1,2]:
    figure_number = numerical_or_analytical
    fig = plt.figure(figure_number)
    if numerical_or_analytical==1:
        fig.suptitle("Analytical Solution")
    else:
        fig.suptitle("Numerical Integration")
    axs1 = [ fig.add_subplot(231), fig.add_subplot(132), fig.add_subplot(133) ] 
    axs2 = [ fig.add_subplot(234), fig.add_subplot(235), fig.add_subplot(236) ] 

    trajectory = numpy.loadtxt(directory+"/demonstration_traj.txt")
    traj_dim0 = trajectory[:,[0,1,3,5]]
    traj_dim1 = trajectory[:,[0,2,4,6]]
    lines = plotTrajectory(traj_dim0,axs1)
    lines.extend(plotTrajectory(traj_dim1,axs2))
    plt.setp(lines, linestyle='-',  linewidth=8, color=(0.4,0.4,0.4))
    
    for goal_number in range(7):
        
        scalings = ["NO_SCALING","G_MINUS_Y0_SCALING","AMPLITUDE_SCALING"]
        for scaling in scalings:
        
            directory_scaling = directory + "/" + scaling 
            #print(directory_scaling)
           
            basename = "reproduced"+str(goal_number)
            if numerical_or_analytical==2:
                basename = "reproduced_num"+str(goal_number)
            trajectory = numpy.loadtxt(directory_scaling+"/"+basename+"_traj.txt")
            traj_dim0 = trajectory[:,[0,1,3,5]]
            traj_dim1 = trajectory[:,[0,2,4,6]]
            lines = plotTrajectory(traj_dim0,axs1)
            lines.extend(plotTrajectory(traj_dim1,axs2))
            
            if scaling=="NO_SCALING":
                plt.setp(lines, linestyle='--', linewidth=4, color=(0.8,0,0))
            if scaling=="G_MINUS_Y0_SCALING":
                plt.setp(lines, linestyle='--', linewidth=3, color=(0,0.7,0))
            if scaling=="AMPLITUDE_SCALING":
                plt.setp(lines, linestyle='-', linewidth=1, color=(0,0.7,0.7))
    
    labels = ["Demonstration"]
    labels.extend(scalings)
    plt.legend(labels)
    

plt.show()
