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


class RichardsNormalizedSystem(DynamicalSystem):
    """ A dynamical system representing a Richard's system (generalized sigmoid system).
    """

    def __init__(self, tau, n_dims, t_inflection_ratio, growth_rate=1.0, v=1.0):
        """ Initialize a RichardsSystem.

        @param tau: Time constant
        @param x_init: Initial state
        """
        super().__init__(1, tau, np.zeros((n_dims,)))
        self._tau = tau  # To avoid flake8 warnings (is already set by super.init above)
        self.t_inflection_ratio = t_inflection_ratio
        self.growth_rate = growth_rate
        self.v = v
        self.right_asymp = np.ones((n_dims,))
        self.left_asymp = None
        self._update_left_asymp()

    def _update_left_asymp(self):
        # Solve for left_asymp, given t_infl, etc.
        t_infl = self.t_inflection_ratio * self.tau
        c = np.power(1.0 + np.exp(self.growth_rate*t_infl), 1.0/self.v)
        self.left_asymp = (c * self.x_init - self.right_asymp) / (c - 1)

    def differential_equation(self, x):
        """ The differential equation which defines the system.

        It relates state values to rates of change of those state values.

        @param x: current state
        @return: xd - rate of change in state
        """

        # https: // en.wikipedia.org / wiki / Generalised_logistic_function
        # Generalised_logistic_differential_equation
        # xd = (B / v) * (1 - ((x - A) / (K - A)) ^ v) * (x - A)
        # B = growth_rate
        # A = left_asymp
        # K = right_asymp
        self._update_left_asymp()
        r = (x-self.left_asymp)/(self.right_asymp - self.left_asymp)
        return (self.growth_rate/self.v) * (1.0 - np.power(r, self.v)) * (x - self.left_asymp) / self.tau

    def analytical_solution(self, ts):
        """
         Return analytical solution of the system at certain times.

         @param ts: A vector of times for which to compute the analytical solutions
         @return: (xs, xds) - Sequence of states and their rates of change.
        """

        xs = np.zeros([ts.size, self._dim_x])
        xds = np.zeros([ts.size, self._dim_x])

        self._update_left_asymp()
        for dd in range(self.dim_x):
            B =  self.growth_rate if np.isscalar(self.growth_rate) else self.growth_rate[dd]
            exp_term = np.exp(-B * ts / self.tau)

            v = self.v if np.isscalar(self.v) else self.v[dd]
            A = self.left_asymp[dd]
            K = self.right_asymp[dd]
            q = -1 + np.power((K-A)/(self.x_init[dd]-A), v)

            xs[:, dd] = (K-A) / np.power(1+q*exp_term, 1/v)
            xs[:, dd] += A

        return xs, xds

    def decouple_parameters(self):
        if np.isscalar(self.t_inflection_ratio):
            self.t_inflection_ratio = np.full((self.dim_x, ), self.t_inflection_ratio)
        if np.isscalar(self.growth_rate):
            self.growth_rate = np.full((self.dim_x, ), self.growth_rate)
        if np.isscalar(self.v):
            self.v = np.full((self.dim_x, ), self.v)
        pass
