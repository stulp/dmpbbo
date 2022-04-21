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
import argparse
import numpy as np
import matplotlib.pyplot as plt

lib_path = os.path.abspath('../python/')
sys.path.append(lib_path)

from performRollouts import *
from TaskThrowBall import *

from to_jsonpickle import *

from bbo.DistributionGaussian import DistributionGaussian

from dmp.dmp_plotting import *
from bbo.bbo_plotting import *


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dmp", help="input dmp")
    parser.add_argument("output_directory", help="directory to write results to")
    parser.add_argument("--sigma", help="sigma of covariance matrix", type=float, default=3.0)
    parser.add_argument("--nsamples", help="number of samples", type=int, default=12)
    parser.add_argument("--show", action='store_true', help="show result plots")
    args = parser.parse_args()

    sigma_dir = 'sigma_%1.3f' % args.sigma
    directory = os.path.join(args.output_directory,sigma_dir)

    filename = args.dmp
    print("Loading DMP from: "+filename)
    with open(filename, 'r') as in_file:
        json = in_file.read()        
        dmp = from_jsonpickle(json)
    
    ts = dmp._ts_train
    ( xs, xds, forcing, fa_outputs) = dmp.analyticalSolution(ts)
    traj_mean = dmp.statesAsTrajectory(ts,xs,xds)
    
    fig = plt.figure(1)
    ax1 = fig.add_subplot(131)
    lines = plotTrajectory(traj_mean.asMatrix(),[ax1])
    plt.setp(lines,linewidth=3)
    
    parameter_vector = dmp.getParameterVectorSelected()
    
    n_samples = args.nsamples
    sigma = args.sigma
    covar_init =  sigma*sigma*np.eye(parameter_vector.size)
    distribution = DistributionGaussian(parameter_vector, covar_init)
    
    filename = os.path.join(directory,f'distribution.json')
    print("Saving sampling distribution to: "+filename)
    os.makedirs(directory,exist_ok=True)
    with open(filename, 'w') as out_file:
        out_file.write(to_jsonpickle(distribution))
    
    samples = distribution.generateSamples(n_samples)

    ax2 = fig.add_subplot(132)
    patch = plot_error_ellipse(distribution.mean[:2],distribution.covar[:2,:2],ax2)
    ax2.plot(samples[:,0],samples[:,1],'o',color='#999999')

    ax3 = fig.add_subplot(133)

    y_floor = -0.3
    x_goal = -0.70
    x_margin = 0.01
    acceleration_weight = 0.001
    task = TaskThrowBall(x_goal,x_margin,y_floor,acceleration_weight)

    for i_sample in range(n_samples):
        
        dmp.setParameterVectorSelected(samples[i_sample,:])

        filename = os.path.join(directory,f'dmp_sample_{i_sample}.json')
        print("Saving sampled DMP to: "+filename)
        with open(filename, 'w') as out_file:
            out_file.write(to_jsonpickle(dmp))

        ( xs, xds, forcing, fa_outputs) = dmp.analyticalSolution()
        traj_sample = dmp.statesAsTrajectory(ts,xs,xds)
        lines = plotTrajectory(traj_sample.asMatrix(),[ax1])
        plt.setp(lines,color='#999999',alpha=0.5)

        cost_vars = performRollouts(dmp,'python_simulation',directory)

        task.plotRollout(cost_vars,ax3)
    
    filename = 'exploration.png'
    fig.savefig(os.path.join(directory,filename))

    
    if args.show:
        plt.show()