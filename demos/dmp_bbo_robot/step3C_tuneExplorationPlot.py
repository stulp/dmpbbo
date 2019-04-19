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


import os
import sys
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Add relative path, in case PYTHONPATH is not set
lib_path = os.path.abspath('../')
sys.path.append(lib_path)
from executeBinary import executeBinary

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from dmp_bbo.dmp_bbo_plotting import *
from dmp.dmp_plotting import *


if __name__=="__main__":

    input_parameters_file = None
    input_task_file = None
    
    if (len(sys.argv)<2):
        print('Usage: '+sys.argv[0]+' <rollouts directory> [task pickle file]')
        print('Example: python3 '+sys.argv[0]+' results/tune_exploration/ results/task.p')
        sys.exit()
        
    if (len(sys.argv)>1):
        directory = sys.argv[1]
    if (len(sys.argv)>2):
        input_task_file = sys.argv[2]
        
    task = None
    if input_task_file:
        task = pickle.load(open(input_task_file, "rb" ))
    
    fig = plt.figure(1)
    n_subplots = 1
    axs = [ fig.add_subplot(1,n_subplots,ii+1) for ii in range(n_subplots) ]
        
    dirs = sorted(glob.glob(directory+"/rollout*"))
    for cur_dir in dirs:
        #plotTrajectoryFromFile(cur_dir+"/cost_vars.txt",axs[1:4])
        if task:
            cost_vars = np.loadtxt(cur_dir+"/cost_vars.txt")
            task.plotRollout(cost_vars,axs[0])

    plt.show()        
    fig.savefig(directory+'/exploration_rollouts.png')
