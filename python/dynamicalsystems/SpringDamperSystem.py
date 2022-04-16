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

import os
import sys

import numpy as np

lib_path = os.path.abspath("../../python/")
sys.path.append(lib_path)

from dynamicalsystems.DynamicalSystem import DynamicalSystem

class SpringDamperSystem(DynamicalSystem):
    def __init__(
        self,
        tau,
        y_init,
        y_attr,
        damping_coefficient,
        spring_constant="CRITICALLY_DAMPED",
        mass=1.0,
    ):
        super().__init__(2, tau, y_init)
        self._y_attr = y_attr
        self._damping_coefficient = damping_coefficient
        self._mass = mass
        if spring_constant == "CRITICALLY_DAMPED":
            self._spring_constant = damping_coefficient * damping_coefficient / 4
        else:
            self._spring_constant = spring_constant

    @property
    def y_attr(self):
        return _y_attr
    
    @y_attr.setter
    def y_attr(self, new_y_attr):
        if new_y_attr.size!=self._dim_y:
            raise ValueError("y_attr must have size "+self._dim_y)
        self._y_attr = np.atleast_1d(new_y_attr)

    def differentialEquation(self, x):

        # Spring-damper system was originally 2nd order, i.e. with [x xd xdd]
        # After rewriting it as a 1st order system it becomes [y z yd zd], with yd = z;
        # Get 'y' and 'z' parts of the state in 'x'
        y = x[0 : self._dim_y]
        z = x[self._dim_y :]

        # Compute yd and zd
        # See  http://en.wikipedia.org/wiki/Damped_spring-mass_system#Example:mass_.E2.80.93spring.E2.80.93damper
        # and equation 2.1 of http://www-clmc.usc.edu/publications/I/ijspeert-NC2013.pdf
        yd = z / self._tau

        zd = (
            -self._spring_constant * (y - self._y_attr) - self._damping_coefficient * z
        ) / (self._mass * self._tau)

        xd = np.concatenate((yd, zd))

        return xd
