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

    directory = None
    if (len(sys.argv)==1):
        print('\nUsage: '+sys.argv[0]+' <directory> [covar_scale] [n_samples]\n')
        sys.exit()
    if (len(sys.argv)>1):
        directory = sys.argv[1]
    
    covar_scale = 1.0
    if (len(sys.argv)>2):
        covar_scale = float(sys.argv[2])
        
    n_samples = 10    
    if (len(sys.argv)>2):
        n_samples = int(sys.argv[3])
        
    print('  * Loading mean from "'+directory+'/parameter_vector_initial.txt"')
    parameter_vector = np.loadtxt(directory+"/parameter_vector_initial.txt")
    covar_init =  covar_scale*np.eye(parameter_vector.size)
    distribution = DistributionGaussian(parameter_vector, covar_init)
    
    
    print('  * Generating '+str(n_samples)+' samples with covar='+str(covar_scale))
    samples = distribution.generateSamples(n_samples)
    #cur_dir = '%s/tune_exploration/rollout%03d' % (directory, i_sample+1)
    
    output_directory = directory+"/tune_exploration/" 
    filename = output_directory + 'samples.txt'  
    print('  * Saving '+str(n_samples)+' samples to '+filename)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    np.savetxt(filename,samples)

    # DMP SAMPLE OUTPUTDIR zzz
    executeBinary("./performDmpRollouts", "results/dmp.xml results/tune_exploration/", True)

    fig = plt.figure(1)
    axs = [ fig.add_subplot(1,2,1), fig.add_subplot(1,3,2), fig.add_subplot(1,3,3)]

    
    for i_sample in range(n_samples):                       
        filename = '%s/traj_sample%05d.txt' % (output_directory, i_sample+1)
        print(filename)
        data = np.loadtxt(filename)
        lines = plotTrajectory(data,axs)
        plt.setp(lines,linestyle='-', linewidth=1, color=(0.2,0.7,0.2), label='perturbed')
        
    filename = output_directory+"/traj_unperturbed.txt"
    print(filename)
    data = np.loadtxt(filename)
    lines = plotTrajectory(data,axs)
    plt.setp(lines,linestyle='-', linewidth=3, color=(0.2,0.2,0.2), label='perturbed')
    #plt.legend()
        

    plt.show()        
