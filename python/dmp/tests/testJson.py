# This file is part of DmpBbo, a set of libraries and programs for the 
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2022 Freek Stulp
# 
# DmpBbo is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# DmpBbo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import os, sys

# Include scripts for plotting
lib_path = os.path.abspath('../../../python/')
sys.path.append(lib_path)

from to_jsonpickle import *

from dmp.Dmp import *
from functionapproximators.FunctionApproximatorRBFN import *
from functionapproximators.FunctionApproximatorLWR import *

if __name__=='__main__':
    """Run some training sessions and plot results."""

    tau = 0.5
    n_dims = 2
    n_time_steps = 51

    y_init = np.linspace(0.0,0.7,n_dims)
    y_attr = np.linspace(0.4,0.5,n_dims)
    
    ts = np.linspace(0,tau,n_time_steps)
    y_yd_ydd_viapoint = np.array([-0.2,0.4, 0.0,0.0, 0,0])
    viapoint_time = 0.4*ts[-1]
    traj = Trajectory.generatePolynomialTrajectoryThroughViapoint(ts, y_init, y_yd_ydd_viapoint, viapoint_time, y_attr)
    

    function_apps = [ FunctionApproximatorRBFN(12,0.7), FunctionApproximatorLWR(10,0.7)]
    dmp = Dmp.from_traj(traj, function_apps)
    
    s = to_jsonpickle(dmp)
    print(s)
    # Save to file
    filename = "Dmp.json"
    with open(filename, "w") as text_file:
        text_file.write(s)


