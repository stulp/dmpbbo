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


class ExponentialSystem(DynamicalSystem):
    def __init__(self, tau, x_init, x_attr, alpha):
        super().__init__(1, tau, x_init)
        self._x_attr = x_attr
        self._alpha = alpha

    @property
    def y_attr(self):
        return _x_attr

    @y_attr.setter
    def y_attr(self, y):
        if y.size != self._dim_y:
            raise ValueError("y_attr must have size " + self._dim_y)
        self._x_attr = np.atleast_1d(y)

    @property
    def x_attr(self):
        return _x_attr

    @x_attr.setter
    def x_attr(self, x):
        if x.size != self._dim_x:
            raise ValueError("y_attr must have size " + self._dim_x)
        self._x_attr = np.atleast_1d(x)

    def differentialEquation(self, x):
        xd = self._alpha * (self._x_attr - x) / self._tau
        return xd

    def analyticalSolution(self, ts):
        T = ts.size

        exp_term = np.exp(-self._alpha * ts / self._tau)
        pos_scale = exp_term
        vel_scale = -(self._alpha / self._tau) * exp_term

        val_range = self._x_init - self._x_attr
        val_range_repeat = np.repeat(np.atleast_2d(val_range), T, axis=0)
        pos_scale_repeat = np.repeat(np.atleast_2d(pos_scale), self._dim_x, axis=0)
        xs = np.multiply(val_range_repeat, pos_scale_repeat.T)

        xs = xs + np.repeat(np.atleast_2d(self._x_attr), T, axis=0)

        vel_scale_repeat = np.repeat(np.atleast_2d(vel_scale), self._dim_x, axis=0)
        xds = np.multiply(val_range_repeat, vel_scale_repeat.T)

        return (xs, xds)
