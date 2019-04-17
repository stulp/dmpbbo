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
import numpy as np
import matplotlib.pyplot as plt

# Add relative path, in case PYTHONPATH is not set
lib_path = os.path.abspath('../')
sys.path.append(lib_path)
from executeBinary import executeBinary

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from bbo.DistributionGaussian import DistributionGaussian
from dmp.dmp_plotting import plotTrajectory


if __name__=="__main__":


    input_dmp_file = None
    input_parameters_file = None
    output_directory = None
    
    if (len(sys.argv)<4):
        print('\nUsage: '+sys.argv[0]+' <input dmp file> <input policy parameters file> <output directory> [covar_scale] [n_samples]\n')
        sys.exit()
        
    if (len(sys.argv)>1):
        input_dmp_file = sys.argv[1]
    if (len(sys.argv)>2):
        input_parameters_file = sys.argv[2]
    if (len(sys.argv)>3):
        output_directory = sys.argv[3]
    
    covar_scale = 1.0
    if (len(sys.argv)>4):
        covar_scale = float(sys.argv[4])
        
    n_samples = 10    
    if (len(sys.argv)>5):
        n_samples = int(sys.argv[5])
        
    print("Python | calling "+" ".join(sys.argv))
    
    print('Python |     Loading mean from "'+input_parameters_file+'"')
    parameter_vector = np.loadtxt(input_parameters_file)
    
    
    print('Python |     Generating '+str(n_samples)+' samples with covar='+str(covar_scale))
    covar_init =  covar_scale*np.eye(parameter_vector.size)
    distribution = DistributionGaussian(parameter_vector, covar_init)
    samples = distribution.generateSamples(n_samples)

    print('Python |     Saving samples to '+output_directory)
    for i_sample in range(n_samples):
        
        rollout_directory = '%s/rollout%03d/' % (output_directory, i_sample+1)
        sample_filename = rollout_directory+'policy_parameters.txt'
        
        print('Python |         Saving sample '+str(i_sample)+' to '+sample_filename)
        if not os.path.exists(rollout_directory):
            os.makedirs(rollout_directory)
        np.savetxt(sample_filename,samples[i_sample,:])
    
        print('Python |         Caling executeBinary ',end="")
        arguments = [input_dmp_file,rollout_directory+"trajectory.txt",sample_filename]
        arguments.append(rollout_directory+"dmp.xml")
        executeBinary("./executeDmp"," ".join(arguments), True)

    fig = plt.figure(1)
    axs = [ fig.add_subplot(1,2,1), fig.add_subplot(1,3,2), fig.add_subplot(1,3,3)]

    
    print("Python |     Plotting")
    for i_sample in range(n_samples):                       
        rollout_directory = '%s/rollout%03d' % (output_directory, i_sample+1)
        traj_filename = rollout_directory+'/trajectory.txt'
        data = np.loadtxt(traj_filename)
        lines = plotTrajectory(data,axs)
        plt.setp(lines,linestyle='-', linewidth=1, color=(0.2,0.2,0.2), label='perturbed')
        
    #filename = output_directory+"/traj_unperturbed.txt"
    #print(filename)
    #data = np.loadtxt(filename)
    #lines = plotTrajectory(data,axs)
    #plt.setp(lines,linestyle='-', linewidth=3, color=(0.2,0.2,0.2), #label='perturbed')
    #plt.legend()
        

    plt.show()        
