# This file is part of DmpBbo, a set of libraries and programs for the
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2023 Freek Stulp
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

    def __init__(self, tau, x_init, x_attr, t_infl_ratio, alpha=1.0, v=1.0):
        super().__init__(1, tau, x_init)
        self._tau = tau  # To avoid flake8 warnings (is already set by super.init above)
        self.alpha = alpha
        self.v = v
        self.right_asymp = x_attr
        self.t_infl_ratio = t_infl_ratio
        self.left_asymp = None

    def set_left_asymp(self, left_asymp):
        self.left_asymp = left_asymp

    def _get_left_asymptote(self):
        if self.left_asymp is not None:
            return self.left_asymp

        t_infl = self.tau * self.t_infl_ratio
        t_infl = np.clip(t_infl, 0.0, None)
        self.v = np.clip(self.v, 0.5, None)
        exp_term = np.exp(-self.alpha * self.v * t_infl / self.tau)
        Z = np.power((self.v / exp_term) + 1, 1 / self.v)
        left_asymp = (Z * self.x_init - self.right_asymp) / (Z - 1)
        return left_asymp

    def differential_equation(self, x):
        """ The differential equation which defines the system.

        It relates state values to rates of change of those state values.

        @param x: current state
        @return: xd - rate of change in state
        """
        left_asymp = self._get_left_asymptote()
        r = (x - left_asymp) / (self.right_asymp - left_asymp)
        return self.alpha * (1.0 - np.power(r, self.v)) * (x - left_asymp) / self.tau

    def analytical_solution(self, ts):
        """
         Return analytical solution of the system at certain times.

         @param ts: A vector of times for which to compute the analytical solutions
         @return: (xs, xds) - Sequence of states and their rates of change.
        """
        xs = np.zeros([ts.size, self._dim_x])
        xds = np.zeros([ts.size, self._dim_x])


        left_asymp = self._get_left_asymptote()
        for dd in range(self.dim_x):
            alpha = self.alpha if np.isscalar(self.alpha) else self.alpha[dd]
            v = self.v if np.isscalar(self.v) else self.v[dd]
            A = left_asymp if np.isscalar(left_asymp) else left_asymp[dd]

            # Giving the variables these names make the relationship to
            # the Wikipedia article clearer.
            # https://en.wikipedia.org/wiki/Generalised_logistic_function
            K = self.right_asymp[dd]
            Q = -1 + np.power((K - A) / (self.x_init[dd] - A), v)

            exp_term = np.exp(-alpha * v * ts / self.tau)

            xs[:, dd] = (K - A) / np.power(1 + Q * exp_term, 1 / v)
            xs[:, dd] += A

            # This is not correct yet
            xds[:, dd] = (Q * alpha * (K - A)) * (exp_term / np.power(1 + Q * exp_term, 1 + 1 / v)) / self.tau

        return xs, xds

    def decouple_parameters(self):
        if np.isscalar(self.t_infl_ratio):
            self.t_infl_ratio = np.full((self.dim_x,), self.t_infl_ratio)
        if np.isscalar(self.alpha):
            self.alpha = np.full((self.dim_x,), self.alpha)
        if np.isscalar(self.v):
            self.v = np.full((self.dim_x,), self.v)
