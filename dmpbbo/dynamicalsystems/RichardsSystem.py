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

    def __init__(self, tau, x_init, left_asymp, right_asymp, growth_rate=1.0, v=1.0):
        """ Initialize a RichardsSystem.

        @param tau: Time constant
        @param x_init: Initial state
        """
        super().__init__(1, tau, x_init)
        self._tau = tau  # To avoid flake8 warnings (is already set by super.init above)
        self.left_asymp = left_asymp
        self.right_asymp = right_asymp
        self.growth_rate = growth_rate
        self.v = v

    @staticmethod
    def as_normalized(tau, t_inflection, growth_rate=1.0, v=1.0):
        """ Initialize a normalized Richards System.

        @param tau: Time constant
        """
        x_0 = np.array([0.0])
        x = np.array([-0.1])
        x_1 = np.array([1.0])
        system = RichardsSystem(tau, x_0, x, x_1, growth_rate, v)
        system.left_asymp = RichardsSystem.compute_left_asymp_from_inflection_time(
            t_inflection, x_0, x_1, growth_rate, v)
        return system

    def differential_equation(self, x):
        """ The differential equation which defines the system.

        It relates state values to rates of change of those state values.

        @param x: current state
        @return: xd - rate of change in state
        """

        # https: // en.wikipedia.org / wiki / Generalised_logistic_function  # Generalised_logistic_differential_equation
        # xd = (B / v) * (1 - ((x - A) / (K - A)) ^ v) * (x - A)
        # B = growth_rate
        # A = left_asymp
        # K = right_asymp
        r = (x-self.left_asymp)/(self.right_asymp-self.left_asymp)
        return (self.growth_rate/self.v) * (1.0 - np.power(r, self.v)) * (x - self.left_asymp) / self.tau

    def analytical_solution(self, ts):
        """
         Return analytical solution of the system at certain times.

         @param ts: A vector of times for which to compute the analytical solutions
         @return: (xs, xds) - Sequence of states and their rates of change.
        """

        xs = np.zeros([ts.size, self._dim_x])
        xds = np.zeros([ts.size, self._dim_x])

        alpha = (self.growth_rate/self.v)
        exp_term = np.exp(-alpha * self.v *  ts / self.tau)

        for dd in range(self.dim_x):
            A = self.left_asymp[dd]
            K = self.right_asymp[dd]
            q = -1 + np.power((K-A)/(self.x_init[dd]-A), self.v)
            xs[:, dd] = (K-A) / np.power(1+q*exp_term, 1/self.v)
            xs[:, dd] += A

        return xs, xds

    @staticmethod
    def compute_left_asymp_from_inflection_time(t_infl, x0, right_asymp, growth_rate, v):
        # Solve for left_asymp, given t_infl, etc.
        c = np.power(1.0 + np.exp(growth_rate*t_infl), 1.0/v)
        left_asymp = (c*x0-right_asymp)/(c-1)
        return left_asymp