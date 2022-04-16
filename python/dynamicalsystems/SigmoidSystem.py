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

from dynamicalsystems.DynamicalSystem import DynamicalSystem  # 

lib_path = os.path.abspath("../../python/")
sys.path.append(lib_path)



class SigmoidSystem(DynamicalSystem):
    def __init__(self, tau, x_init, max_rate, inflection_point_time):
        super().__init__(1, tau, x_init)
        self.max_rate_ = max_rate
        self.inflection_point_time_ = inflection_point_time

        self.Ks_ = self.computeKs(x_init, max_rate, inflection_point_time)

    def set_tau(self, new_tau):
        # Get previous tau from superclass with tau() and set it with set_tau()
        prev_tau = self.tau_
        super().set_tau(new_tau)

        self.inflection_point_time_ = new_tau * self.inflection_point_time_ / prev_tau
        self.Ks_ = self.computeKs(
            self.x_init_, self.max_rate_, self.inflection_point_time_
        )

    def set_x_init(self, x_init):
        assert x_init.size() == dim_x_
        super().set_x_init(x_init)
        self.Ks_ = self.computeKs(x_init_, max_rate_, inflection_point_time_)

    def computeKs(self, N_0s, r, inflection_point_time_time):
        # The idea here is that the initial state (called N_0s above), max_rate (r above) and the
        # inflection_point_time are set by the user.
        # The only parameter that we have left to tune is the "carrying capacity" K.
        #   http://en.wikipedia.org/wiki/Logistic_function#In_ecology:_modeling_population_growth
        # In the below, we set K so that the initial state is N_0s for the given r and tau

        # Known
        #   N(t) = K / ( 1 + (K/N_0 - 1)*exp(-r*t))
        #   N(t_inf) = K / 2
        # Plug into each other and solve for K
        #   K / ( 1 + (K/N_0 - 1)*exp(-r*t_infl)) = K/2
        #              (K/N_0 - 1)*exp(-r*t_infl) = 1
        #                             (K/N_0 - 1) = 1/exp(-r*t_infl)
        #                                       K = N_0*(1+(1/exp(-r*t_infl)))
        Ks = np.empty(N_0s.shape)
        for dd in range(len(Ks)):
            Ks[dd] = N_0s[dd] * (1.0 + (1.0 / np.exp(-r * inflection_point_time_time)))

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
        div = np.divide(N_0s, Ks) - 1.0
        if np.any(np.abs(div) < 10e-9):  # 10e-9 determined empirically
            print(
                "In function SigmoidSystem::computeKs(), Ks is too close to N_0s. This may lead to errors during numerical integration. Recommended solution: choose a lower magnitude for the maximum rate of change (currently it is "
                + str(r)
                + ")"
            )

        return Ks

    def differentialEquation(self, x):
        xd = self.max_rate_ * x * (1 - (np.divide(x, self.Ks_)))
        return xd

    def analyticalSolutionToFix(self, ts):
        # Auxillary variables to improve legibility
        r = self.max_rate_
        exp_rt = np.exp(-r * ts)

        xs = np.empty([ts.size, self.dim_x_])
        xds = np.empty([ts.size, self.dim_x_])

        print(xs.shape)
        print(xds.shape)
        print(exp_rt.shape)

        for dd in range(self.dim_x_):
            # Auxillary variables to improve legibility
            K = self.Ks_[dd]
            b = (K / self.x_init_[dd]) - 1
            print(K.shape)
            print(b.shape)

            xs[:, dd] = K / (1 + b * exp_rt)
            xds[:, dd] = np.multiply((K * r * b) / np.square(1.0 + b * exp_rt), exp_rt)

        return (xs, xds)
