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

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from dmp_bbo.dmp_bbo_plotting import *
from dmp.dmp_plotting import *


if __name__=="__main__":

    input_cost_vars_file = None
    input_task_file = None
    
    if len(sys.argv)<3:
        print('Usage: '+sys.argv[0]+' <cost vars file> <task pickle file>')
        print('Example: python3 '+sys.argv[0]+' results/cost_vars_demonstrated.txt results/task.p')
        sys.exit()
        
    if (len(sys.argv)>1):
        input_cost_vars_file = sys.argv[1]
    if (len(sys.argv)>2):
        input_task_file = sys.argv[2]
        
    task = None
    if input_task_file:
        task = pickle.load(open(input_task_file, "rb" ))
    
    fig = plt.figure(1)
    n_subplots = 1
    axs = [ fig.add_subplot(1,n_subplots,ii+1) for ii in range(n_subplots) ]
        
    cost_vars = np.loadtxt(input_cost_vars_file)
    task.plotRollout(cost_vars,axs[0])

    plt.show()
