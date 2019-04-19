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


import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

lib_path = os.path.abspath('../../python')
sys.path.append(lib_path)

from bbo.updaters import *
from bbo.DistributionGaussian import DistributionGaussian

from dmp_bbo.Task import Task
from dmp_bbo.run_one_update import runOptimizationTaskOneUpdate

if __name__=="__main__":

    # See if input directory was passed
    if (len(sys.argv)>=2):
        directory = str(sys.argv[1])
    else:
        print('\nUsage: '+sys.argv[0]+' <directory>\n')
        sys.exit()

    ############################################################
    # TUNE OPTIMIZATION ALGORITHM PARAMETERS HERE


    eliteness = 10
    weighting_method = 'PI-BB'
    updater_mean = UpdaterMean(eliteness,weighting_method)
    
    covar_decay_factor = 0.8
    updater_decay = UpdaterCovarDecay(eliteness,weighting_method,covar_decay_factor)
    
    min_level = 0.1
    max_level = 10.0
    diag_only = True
    learning_rate=0.5
    #updater = UpdaterCovarAdaptation(eliteness, weighting_method)
    updater_adaptation = UpdaterCovarAdaptation(eliteness, weighting_method,max_level,min_level,diag_only,learning_rate)
    
    updater = updater_adaptation
    
    n_samples_per_update = 10
    
    ############################################################

    # Load the initial distribution
    input_mean_file = directory+'/policy_parameters.txt'
    input_covar_file = directory+'/distribution_initial_covar.txt'
    print('Python |     Loading distribution from "'+input_mean_file+'" and "'+input_covar_file+'"')
    mean_init = np.loadtxt(input_mean_file)
    covar_init = np.loadtxt(input_covar_file)
    initial_distribution = DistributionGaussian(mean_init, covar_init)

    # Load the task
    task = pickle.load(open(directory+'/task.p', "rb" ))
    
    # Execute one update
    i_update = runOptimizationTaskOneUpdate(directory, task, initial_distribution, updater, n_samples_per_update)
    
    #print("./robotPerformRollouts.bash $DIRECTORY/dmp.xml ${UPDATE_DIR}")
