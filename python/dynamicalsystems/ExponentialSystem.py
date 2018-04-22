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

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)

from dynamicalsystems.DynamicalSystem import DynamicalSystem# 


class ExponentialSystem(DynamicalSystem):
    
    def __init__(self, tau, y_init, y_attr, alpha, name="ExponentialSystem"):
        super().__init__(1, tau, y_init, y_attr, name)
        self.alpha_ = alpha

    def differentialEquation(self, x):
        xd = self.alpha_*(self.attractor_state_-x)/self.tau_
        return xd

    def analyticalSolution(self, ts):
        T = ts.size

        exp_term  = np.exp(-self.alpha_*ts/self.tau_);
        pos_scale = exp_term;
        vel_scale = -(self.alpha_/self.tau_) * exp_term;
        
        val_range = self.initial_state_ - self.attractor_state_;
        val_range_repeat = np.repeat(np.atleast_2d(val_range),T,axis=0)
        pos_scale_repeat = np.repeat(np.atleast_2d(pos_scale),self.dim_,axis=0) 
        xs = np.multiply(val_range_repeat,pos_scale_repeat.T)
        
        attr_repeat = np.repeat(np.atleast_2d(self.attractor_state_),T,axis=0)
        xs = xs + attr_repeat
        
        vel_scale_repeat = np.repeat(np.atleast_2d(vel_scale),self.dim_,axis=0) 
        xds = np.multiply(val_range_repeat,vel_scale_repeat.T)

        return (xs, xds)