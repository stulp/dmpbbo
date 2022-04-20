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
    n_time_steps = 51

    y_init = np.linspace(0.0,0.7,n_dims)
    y_attr = np.linspace(0.4,0.5,n_dims)
    
    ts = np.linspace(0,tau,n_time_steps)
    y_yd_ydd_viapoint = np.array([-0.2,0.4, 0.0,0.0, 0,0])
    viapoint_time = 0.4*ts[-1]
    traj = Trajectory.generatePolynomialTrajectoryThroughViapoint(ts, y_init, y_yd_ydd_viapoint, viapoint_time, y_attr)
    

    function_apps = [ FunctionApproximatorRBFN(10,0.7) for i_dim in range(n_dims) ]
    #dmp_type='IJSPEERT_2002_MOVEMENT'
    dmp_type='KULVICIUS_2012_JOINING'
    #dmp_type='COUNTDOWN_2013'
    dmp = Dmp.from_traj(traj, function_apps, dmp_type)


    tau_exec = 0.7
    n_time_steps = 71
    ts = np.linspace(0,tau_exec,n_time_steps)
    
    ( xs_ana, xds_ana, forcing_terms_ana, fa_outputs_ana) = dmp.analyticalSolution(ts)

    dt = ts[1]
    xs_step = np.zeros([n_time_steps,dmp._dim_x])
    xds_step = np.zeros([n_time_steps,dmp._dim_x])
    
    (x,xd) = dmp.integrateStart()
    xs_step[0,:] = x;
    xds_step[0,:] = xd;
    for tt in range(1,n_time_steps):
        (xs_step[tt,:],xds_step[tt,:]) = dmp.integrateStep(dt,xs_step[tt-1,:]); 

    print("Plotting")
    
    fig = plt.figure(1,figsize=(15,9))
    plotDmp(tau,ts,xs_ana,xds_ana,fig,forcing_terms_ana,fa_outputs_ana)
    fig.canvas.set_window_title('Analytical integration') 
    
    fig = plt.figure(2,figsize=(15,9))
    plotDmp(tau,ts,xs_step, xds_step,fig)
    fig.canvas.set_window_title('Step-by-step integration') 
    
    
    plt.show()
