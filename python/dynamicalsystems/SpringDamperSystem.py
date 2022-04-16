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

from dynamicalsystems.DynamicalSystem import DynamicalSystem


class SpringDamperSystem(DynamicalSystem):
    
    def __init__(self, tau, y_init, y_attr, damping_coefficient, spring_constant="CRITICALLY_DAMPED", mass=1.0):
        super().__init__(2, tau, y_init)
        self.y_attr_ = y_attr
        self.damping_coefficient_ = damping_coefficient
        self.mass_ = mass
        if spring_constant == "CRITICALLY_DAMPED":
            self.spring_constant_ = damping_coefficient*damping_coefficient/4
        else:
            self.spring_constant_ = spring_constant

    def set_y_attr(self,y_attr):
        y_attr_ = y_attr
        
    def differentialEquation(self, x):
        
        # Spring-damper system was originally 2nd order, i.e. with [x xd xdd]
        #After rewriting it as a 1st order system it becomes [y z yd zd], with yd = z; 
        # Get 'y' and 'z' parts of the state in 'x'
        y = x[0:self.dim_y_]
        z = x[self.dim_y_:]
        
        # Compute yd and zd
        # See  http://en.wikipedia.org/wiki/Damped_spring-mass_system#Example:mass_.E2.80.93spring.E2.80.93damper
        # and equation 2.1 of http://www-clmc.usc.edu/publications/I/ijspeert-NC2013.pdf
        yd = z/self.tau_;
  
        zd = (-self.spring_constant_*(y-self.y_attr_) - self.damping_coefficient_*z)/(self.mass_*self.tau_)
        
        xd = np.concatenate((yd,zd))
  
        return xd
