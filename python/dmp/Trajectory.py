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
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
# 

import numpy as np
import sys
import os

#lib_path = os.path.abspath('../../python/dynamicalsystems')
#sys.path.append(lib_path)


class Trajectory:

    def __init__(self,  ts, ys, yds, ydds, misc=None):
        n_time_steps = ts.size
        assert(n_time_steps==ys.shape[0])
        assert(n_time_steps==yds.shape[0])
        assert(n_time_steps==ydds.shape[0])
        if misc:
            assert(n_time_steps==misc_.shape[0])
            
        self.dim_ = 1
        if ys.ndim==2:
            self.dim_ = ys.shape[1]
            
        self.ts_ = ts
        self.ys_ = ys
        self.yds_ = yds
        self.ydds_ = ydds
        self.misc_ = misc

    def generatePolynomialTrajectory(ts, y_from, yd_from, ydd_from, y_to, yd_to, ydd_to):
        
        a0 = y_from
        a1 = yd_from
        a2 = ydd_from / 2

        a3 = -10 * y_from - 6 * yd_from - 2.5 * ydd_from + 10 * y_to - 4 * yd_to + 0.5 * ydd_to
        a4 = 15 * y_from + 8 * yd_from + 2 * ydd_from - 15  * y_to  + 7 * yd_to - ydd_to
        a5 = -6 * y_from - 3 * yd_from - 0.5 * ydd_from  + 6 * y_to  - 3 * yd_to + 0.5 * ydd_to

        n_time_steps = ts.size
        n_dims = y_from.size
  
        ys = np.zeros([n_time_steps,n_dims])
        yds = np.zeros([n_time_steps,n_dims])
        ydds = np.zeros([n_time_steps,n_dims])

        for i in range(n_time_steps):
            t = (ts[i] - ts[0]) / (ts[n_time_steps - 1] - ts[0])
            ys[i,:] = a0 + a1 * t + a2 * pow(t, 2) + a3 * pow(t, 3) + a4 * pow(t, 4) + a5 * pow(t, 5)
            yds[i,:] = a1 + 2 * a2 * t + 3 * a3 * pow(t, 2) + 4 * a4 * pow(t, 3) + 5 * a5 * pow(t, 4)
            ydds[i,:] = 2 * a2 + 6 * a3 * t + 12 * a4 * pow(t, 2) + 20 * a5 * pow(t, 3)

        yds /= (ts[n_time_steps - 1] - ts[0])
        ydds /= pow(ts[n_time_steps - 1] - ts[0], 2)

        return Trajectory(ts, ys, yds, ydds)

    def generatePolynomialTrajectoryThroughViapoint(ts, y_from, y_yd_ydd_viapoint, viapoint_time, y_to):
        
        n_time_steps = ts.size
        n_dims = y_from.size
  
        assert(n_dims==y_to.size)
        assert(3*n_dims==y_yd_ydd_viapoint.size)# Contains y, yd and ydd, so *3
  
        viapoint_time_step = 0
        while viapoint_time_step<n_time_steps and ts[viapoint_time_step]<viapoint_time:
            viapoint_time_step+=1
  
        #if (viapoint_time_step>=n_time_steps)
        #{
        #cerr << __FILE__ << ":" << __LINE__ << ":"
        #cerr << "ERROR: the time vector does not contain any time smaller than " << viapoint_time << ". Returning min-jerk trajectory WITHOUT viapoint." <<  endl
        #return Trajectory()
        #}

        yd_from        = np.zeros(n_dims)
        ydd_from       = np.zeros(n_dims)
        
        y_viapoint     = y_yd_ydd_viapoint[0*n_dims:1*n_dims]
        yd_viapoint    = y_yd_ydd_viapoint[1*n_dims:2*n_dims]
        ydd_viapoint   = y_yd_ydd_viapoint[2*n_dims:3*n_dims]
        
        yd_to          = np.zeros(n_dims)
        ydd_to         = np.zeros(n_dims)

        traj1 = Trajectory.generatePolynomialTrajectory(ts[:viapoint_time_step], y_from, yd_from, ydd_from, y_viapoint, yd_viapoint, ydd_viapoint)

        traj2 = Trajectory.generatePolynomialTrajectory(ts[viapoint_time_step:], y_viapoint, yd_viapoint, ydd_viapoint, y_to, yd_to, ydd_to)
        
        traj1.append(traj2)
        
        return traj1


    def generateMinJerkTrajectory(ts, y_from, y_to):
        n_time_steps = ts.size
        n_dims = y_from.size
        
        ys = np.zeros([n_time_steps,n_dims])
        yds = np.zeros([n_time_steps,n_dims])
        ydds = np.zeros([n_time_steps,n_dims])
  
        D =  ts[n_time_steps-1]
        tss = ts/D 

        A = y_to-y_from
  
        for i_dim in range(n_dims):
            
            #http://noisyaccumulation.blogspot.fr/2012/02/how-to-decompose-2d-trajectory-data.html
    
            ys[:,i_dim]   = y_from[i_dim] + A[i_dim]*(  6*(tss**5)  -15*(tss**4) +10*(tss**3))
   
            yds[:,i_dim]  =             (A[i_dim]/D)*( 30*(tss**4)  -60*(tss**3) +30*(tss**2))
    
            ydds[:,i_dim] =         (A[i_dim]/(D*D))*(120*(tss**3) -180*(tss**2) +60*tss       )

        return Trajectory(ts,ys,yds,ydds)
        
    def append(self,trajectory):
        self.ts_ = np.concatenate((self.ts_, trajectory.ts_))
        self.ys_ = np.concatenate((self.ys_, trajectory.ys_))
        self.yds_ = np.concatenate((self.yds_, trajectory.yds_))
        self.ydds_ = np.concatenate((self.ydds_, trajectory.ydds_))
        if self.misc_ is None or trajectory.misc_ is None:
            self.misc_ = None
        else:
            self.misc_ = np.concatenate((self.misc_, trajectory.misc_))
        
    def asMatrix(self):
        as_matrix = np.column_stack((self.ts_, self.ys_, self.yds_, self.ydds_))
        if self.misc_:
            np.column_stack((as_matrix,self.misc_))
        return as_matrix

        