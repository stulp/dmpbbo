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

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from bbo.DistributionGaussian import DistributionGaussian


if __name__=="__main__":

    input_parameters_file = None
    output_covar_file = None
    output_directory = None
    
    if (len(sys.argv)<3):
        print('Usage: '+sys.argv[0]+' <input policy parameters file> <output covar file> <output directory> [covar_scale] [n_samples]')
        print('Example: python3 '+sys.argv[0]+' results/policy_parameters.txt results/distribution_initial_covar.txt results/tune_exploration/ 10.0 12')
        sys.exit()
        
    if (len(sys.argv)>1):
        input_parameters_file = sys.argv[1]
    if (len(sys.argv)>2):
        output_covar_file = sys.argv[2]
    if (len(sys.argv)>3):
        output_directory = sys.argv[3]
    
    covar_scale = 1.0
    if (len(sys.argv)>3):
        covar_scale = float(sys.argv[4])
        
    n_samples = 10    
    if (len(sys.argv)>4):
        n_samples = int(sys.argv[5])
        
    print("Python | calling "+" ".join(sys.argv))
    
    print('Python |     Loading mean from "'+input_parameters_file+'"')
    parameter_vector = np.loadtxt(input_parameters_file)
    
    
    print('Python |     Generating '+str(n_samples)+' samples with covar='+str(covar_scale))
    covar_init =  covar_scale*np.eye(parameter_vector.size)
    distribution = DistributionGaussian(parameter_vector, covar_init)
    samples = distribution.generateSamples(n_samples)

    print('Python |     Saving covar to "'+output_covar_file+'"')
    np.savetxt(output_covar_file,covar_init)
    
    print('Python |     Saving samples to '+output_directory)
    for i_sample in range(n_samples):
        
        rollout_directory = '%s/rollout%03d/' % (output_directory, i_sample+1)
        sample_filename = rollout_directory+'sample.txt'
        
        print('Python |         Saving sample '+str(i_sample)+' to '+sample_filename)
        if not os.path.exists(rollout_directory):
            os.makedirs(rollout_directory)
        np.savetxt(sample_filename,samples[i_sample,:])

