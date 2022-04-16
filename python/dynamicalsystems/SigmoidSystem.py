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

from dynamicalsystems.DynamicalSystem import DynamicalSystem  # 

class SigmoidSystem(DynamicalSystem):
    def __init__(self, tau, x_init, max_rate, inflection_point_time):
        super().__init__(1, tau, x_init)
        self._max_rate = max_rate
        self._inflection_point_time = inflection_point_time
        self._Ks_cached = None

    @DynamicalSystem.tau.setter
    def tau(self, new_tau):
        if hasattr(self, '_tau'):
            self._inflection_point_time *= new_tau / self._tau
        self.tau_ = new_tau
        self._Ks_cached = None

    @DynamicalSystem.x_init.setter
    def x_init(self, new_x_init):
        super().x_init = new_x_init
        self._Ks_cached = None
        
    def _Ks(self):
        if isinstance(self._Ks_cached,np.ndarray):
            # Cached variable is available
            return self._Ks_cached
        
        # The idea here is that the initial state (called N_0s above), max_rate (r above) and the
        # inflection_point_time are set by the user.
        # The only parameter that we have left to tune is the "carrying capacity" K.
        #   http://en.wikipedia.org/wiki/Logistic_function#In_ecology:_modeling_population_growth
        # In the below, we set K so that the initial state is N_0s for the given r and tau

        N_0s = self._x_init
        r = self._max_rate 
        t_infl = self._inflection_point_time
        
        # Known
        #   N(t) = K / ( 1 + (K/N_0 - 1)*exp(-r*t))
        #   N(t_inf) = K / 2
        # Plug into each other and solve for K
        #   K / ( 1 + (K/N_0 - 1)*exp(-r*t_infl)) = K/2
        #              (K/N_0 - 1)*exp(-r*t_infl) = 1
        #                             (K/N_0 - 1) = 1/exp(-r*t_infl)
        #                                       K = N_0*(1+(1/exp(-r*t_infl)))
        self._Ks_cached = np.empty(N_0s.shape)
        for dd in range(len(N_0s)):
            self._Ks_cached[dd] = N_0s[dd] * (1.0 + (1.0 / np.exp(-r * t_infl)))

        # If Ks is too close to N_0===initial_state, then the differential equation will always return 0
        # See differentialEquation below
        #   xd = max_rate_*x*(1-(x/Ks_))
        # For initial_state this is
        #   xd = max_rate_*initial_state*(1-(initial_state/Ks_))
        # If initial_state is very close/equal to Ks we get
        #   xd = max_rate_*Ks*(1-(Ks/Ks_))
        #   xd = max_rate_*Ks*(1-1)
        #   xd = max_rate_*Ks*0
        #   xd = 0
        # And integration fails, especially for Euler integration.
        # So we now give a warning if this is likely to happen.
        div = np.divide(N_0s, self._Ks_cached) - 1.0
        if np.any(np.abs(div) < 10e-9):  # 10e-9 determined empirically
            print(
                "In function SigmoidSystem::computeKs(), Ks is too close to N_0s. This may lead to errors during numerical integration. Recommended solution: choose a lower magnitude for the maximum rate of change (currently it is "
                + str(r)
                + ")"
            )

        return self._Ks_cached

    def differentialEquation(self, x):
        xd = self._max_rate * x * (1 - (np.divide(x, self._Ks())))
        return xd

    def analyticalSolutionToFix(self, ts):
        # Auxillary variables to improve legibility
        r = self._max_rate
        exp_rt = np.exp(-r * ts)

        xs = np.empty([ts.size, self._dim_x])
        xds = np.empty([ts.size, self._dim_x])

        print(xs.shape)
        print(xds.shape)
        print(exp_rt.shape)

        Ks = self._Ks()
        for dd in range(self._dim_x):
            # Auxillary variables to improve legibility
            b = (Ks[dd] / self._x_init[dd]) - 1
            print(K.shape)
            print(b.shape)

            xs[:, dd] = K / (1 + b * exp_rt)
            xds[:, dd] = np.multiply((K * r * b) / np.square(1.0 + b * exp_rt), exp_rt)

        return (xs, xds)
