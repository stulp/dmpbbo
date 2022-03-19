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


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys
import argparse
import pickle

lib_path = os.path.abspath('../python/')
sys.path.append(lib_path)

from dmp.dmp_plotting import *
from dmp.Dmp import *
from dmp.Trajectory import *
from functionapproximators.FunctionApproximatorRBFN import *
from functionapproximators.FunctionApproximatorLWR import *
from functionapproximators.functionapproximators_plotting import * 
from dmp.dmp_plotting import * 

def legendWithoutDuplicates(ax):
    handles, labels = ax.get_legend_handles_labels()  
    lgd = dict(zip(labels, handles))
    ax.legend(lgd.values(), lgd.keys())
#    plt.legend()


if __name__=='__main__':
  
    parser = argparse.ArgumentParser()
    parser.set_defaults(use_offset=False)
    parser.add_argument("input_trajectory_file", type=str, help="input trajectory file")
    parser.add_argument("output_dir", type=str, help="output directory to write data from")
    parser.add_argument("--n", help="Number of basis functions",default=5)
    args = parser.parse_args()
  
    n_basis_functions = args.n
    directory = args.output_dir
    
    trajectory = Trajectory.readFromFile(args.input_trajectory_file)

    ts = trajectory.ts_;
    n_dims = trajectory.dim();

    function_approximators = []
    input_dim = 1;
    intersection = 0.7;
    for dd in range(n_dims):
        fa = FunctionApproximatorRBFN(input_dim,n_basis_functions,intersection)
        # Set the parameters to optimize
        fa.setSelectedParameters(['weights'])
        function_approximators.append(fa)
    
  
    # Initialize the DMP
    tau = trajectory.ts_[-1]
    y_init = trajectory.ys_[0,:]
    y_attr = trajectory.ys_[-1,:]
    dmp = Dmp(tau, y_init, y_attr, function_approximators, "trained",-12)
    dmp.train(trajectory)

    # Set which parameters to optimize
    #dmp->setSelectedParameters(parameters_to_optimize);

    pickle.dump(dmp, open(directory+'/dmp.p', "wb" ))
    
    # Save the initial parameter vector to file
    #Eigen::VectorXd parameter_vector;
    #dmp->getParameterVectorSelected(parameter_vector);
    # overwrite = true;
    #cout << "C++    |     Writing initial parameter vector to file : " << output_parameters_file << endl;
    #saveMatrix(output_parameters_file,parameter_vector,overwrite);
  
    ( xs_ana, xds_ana, forcing_terms_ana, fa_outputs_ana) = dmp.analyticalSolution(ts)
    traj_reproduced = dmp.statesAsTrajectory(ts,xs_ana,xds_ana)
    
    # Plot trajectories    
    golden_ratio = 1.618 # graphs have nice proportions with this ratio
    n_subplots = 3
    fig3 = plt.figure(3,figsize=(golden_ratio*n_subplots*3,3))
    axs = [ fig3.add_subplot(1,n_subplots,ii+1) for ii in range(n_subplots) ]
    
    traj_demonstration = trajectory
    lines = plotTrajectory(traj_demonstration.asMatrix(),axs)
    plt.setp(lines, linestyle='-',  linewidth=4, color=(0.8,0.8,0.8), label='demonstration')
    
    lines = plotTrajectory(traj_reproduced.asMatrix(),axs)
    plt.setp(lines, linestyle='--', linewidth=2, color=(0.0,0.0,0.5), label='reproduced')
    
    legendWithoutDuplicates(axs[-1])
    
    fig3.canvas.set_window_title('Comparison between demonstration and reproduced') 
    plt.tight_layout()
    
    plt.show()
