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
#
""" Module for the RichardsSystem class. """

import numpy as np

from dmpbbo.dynamicalsystems.DynamicalSystem import DynamicalSystem


class RichardsSystem(DynamicalSystem):
    """ A dynamical system representing a Richard's system (generalized sigmoid system).
    """

    def __init__(self, tau, x_init, x_attr, t_infl, alpha=1.0, v=1.0):
        super().__init__(1, tau, x_init)
        self._tau = tau  # To avoid flake8 warnings (is already set by super.init above)
        self.alpha = alpha
        self.v = v
        self.right_asymp = x_attr
        self.left_asymp = None
        self.set_t_infl(t_infl)

    def set_left_asymp(self, left_asymp):
        self.left_asymp = left_asymp

    def set_t_infl(self, t_infl):
        exp_term = np.exp(-self.alpha * self.v * t_infl / self.tau)
        Z = np.power((self.v / exp_term) + 1, 1 / self.v)
        self.left_asymp = (Z * self.x_init - self.right_asymp) / (Z - 1)


    def differential_equation(self, x):
        r = (x - self.left_asymp) / (self.right_asymp - self.left_asymp)
        return self.alpha * (1.0 - np.power(r, self.v)) * (x - self.left_asymp) / self.tau

    def analytical_solution(self, ts):
        xs = np.zeros([ts.size, self._dim_x])
        xds = np.zeros([ts.size, self._dim_x])

        exp_term = np.exp(-self.alpha * self.v * ts / self.tau)

        # Giving the variables these names make the relationship to
        # the Wikipedia article clearer.
        # https://en.wikipedia.org/wiki/Generalised_logistic_function
        A = self.left_asymp
        K = self.right_asymp
        Q = -1 + np.power((K - A) / (self.x_init - A), self.v)

        xs = (K - A) / np.power(1 + Q * exp_term, 1 / self.v)
        xs += A

        # This is not correct yet
        xds = (Q * self.alpha * (K - A)) * (exp_term / np.power(1 + Q * exp_term, 1 + 1 / self.v))

        return xs, xds

    def decouple_parameters(self):
        if np.isscalar(self.t_inflection_ratio):
            self.t_inflection_ratio = np.full((self.dim_x, ), self.t_inflection_ratio)
        if np.isscalar(self.growth_rate):
            self.growth_rate = np.full((self.dim_x, ), self.growth_rate)
        if np.isscalar(self.v):
            self.v = np.full((self.dim_x, ), self.v)
