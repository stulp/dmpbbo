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

lib_path = os.path.abspath("../../python/")
sys.path.append(lib_path)

from dynamicalsystems.DynamicalSystem import DynamicalSystem  #


class ExponentialSystem(DynamicalSystem):
    def __init__(self, tau, x_init, x_attr, alpha):
        super().__init__(1, tau, x_init)
        self.x_attr_ = x_attr
        self.alpha_ = alpha

    def set_y_attr(self, y_attr):
        self.x_attr_ = np.atleast_1d(y_attr)

    def set_x_attr(self, x_attr):
        self.x_attr_ = np.atleast_1d(x_attr)

    def differentialEquation(self, x):
        xd = self.alpha_ * (self.x_attr_ - x) / self.tau_
        return xd

    def analyticalSolution(self, ts):
        T = ts.size

        exp_term = np.exp(-self.alpha_ * ts / self.tau_)
        pos_scale = exp_term
        vel_scale = -(self.alpha_ / self.tau_) * exp_term

        val_range = self.x_init_ - self.x_attr_
        val_range_repeat = np.repeat(np.atleast_2d(val_range), T, axis=0)
        pos_scale_repeat = np.repeat(np.atleast_2d(pos_scale), self.dim_x_, axis=0)
        xs = np.multiply(val_range_repeat, pos_scale_repeat.T)

        xs = xs + np.repeat(np.atleast_2d(self.x_attr_), T, axis=0)

        vel_scale_repeat = np.repeat(np.atleast_2d(vel_scale), self.dim_x_, axis=0)
        xds = np.multiply(val_range_repeat, vel_scale_repeat.T)

        return (xs, xds)
