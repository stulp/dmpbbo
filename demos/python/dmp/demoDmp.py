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

lib_path = os.path.abspath('../../../python/')
sys.path.append(lib_path)

from dmp.dmp_plotting import *
from dmp.Dmp import *
from functionapproximators.FunctionApproximatorRBFN import *

if __name__=='__main__':

    tau = 0.5
    n_dims = 2
    y_init = np.linspace(0.0,0.7,n_dims)
    y_attr = np.linspace(0.4,0.5,n_dims)

    function_apps = []
    for n_basis in [5,6]:
        
        fa = FunctionApproximatorRBFN(n_basis)
        
        n_samples = 100
        fa.train(np.linspace(0,1,n_samples),np.zeros(n_samples))
        
        fa.setSelectedParameters('weights')
        random_weights = 10*np.random.normal(0,1,n_basis)
        fa.setParameterVectorSelected(random_weights)
        
        function_apps.append(fa)

        
    dmp = Dmp(tau, y_init, y_attr, function_apps)

    tau_exec = 0.7
    n_time_steps = 71
    ts = np.linspace(0,tau_exec,n_time_steps)
    
    ( xs_ana, xds_ana, forcing_terms_ana, fa_outputs_ana) = dmp.analyticalSolution(ts)

    dt = ts[1]
    xs_step = np.zeros([n_time_steps,dmp.dim_])
    xds_step = np.zeros([n_time_steps,dmp.dim_])
    
    (x,xd) = dmp.integrateStart()
    xs_step[0,:] = x;
    xds_step[0,:] = xd;
    for tt in range(1,n_time_steps):
        (xs_step[tt,:],xds_step[tt,:]) = dmp.integrateStep(dt,xs_step[tt-1,:]); 

    print("Plotting")
    
    #fig = plt.figure(1)
    #axs = [ fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133) ] 
    #
    #lines = plotTrajectoryFromFile(directory+"/demonstration_traj.txt",axs)
    #plt.setp(lines, linestyle='-',  linewidth=4, color=(0.8,0.8,0.8), label='demonstration')
    #
    #lines = plotTrajectoryFromFile(directory+"/reproduced_traj.txt",axs)
    #plt.setp(lines, linestyle='--', linewidth=2, color=(0.0,0.0,0.5), label='reproduced')
    #
    #plt.legend()
    #fig.canvas.set_window_title('Comparison between demonstration and reproduced') 
    #
    ## Read data
    #xs_xds        = numpy.loadtxt(directory+'/reproduced_xs_xds.txt')
    #forcing_terms = numpy.loadtxt(directory+'/reproduced_forcing_terms.txt')
    #fa_output     = numpy.loadtxt(directory+'/reproduced_fa_output.txt')
    
    fig = plt.figure(2)
    ts_xs_xds = np.column_stack((ts,xs_ana,xds_ana))
    plotDmp(ts_xs_xds,fig,forcing_terms_ana,fa_outputs_ana)
    fig.canvas.set_window_title('Analytical integration') 
    
    fig = plt.figure(3)
    ts_xs_xds = np.column_stack((ts,xs_step,xds_step))
    plotDmp(ts_xs_xds,fig)
    fig.canvas.set_window_title('Step-by-step integration') 
    
    
    plt.show()
