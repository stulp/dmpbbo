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
""" Module for the SigmoidSystem class. """

import numpy as np

from dmpbbo.dynamicalsystems.DynamicalSystem import DynamicalSystem


class SigmoidSystem(DynamicalSystem):
    """ A dynamical system representing a sigmoid system.
    """

    def __init__(self, tau, x_init, max_rate, inflection_ratio):
        """ Initialize a SigmoidSystem.

        @param tau: Time constant
        @param x_init: Initial state
        @param max_rate: Maximum rate of change
        @param inflection_ratio: Time at which maximum rate of change is
                               achieved, i.e. at inflection_ratio * tau
        """
        super().__init__(1, tau, x_init)
        self._tau = tau  # To avoid flake8 warnings (is already set by super.init above)
        self._max_rate = max_rate
        if isinstance(self._max_rate, list):
            self._max_rate = np.asarray(self._max_rate)
        self._inflection_ratio = inflection_ratio
        if isinstance(self._inflection_ratio, list):
            self._inflection_ratio = np.asarray(self._inflection_ratio)
        self._Ks_cached = None

    @classmethod
    def for_gating(cls, tau, y_tau_0_ratio=0.1, n_dims=1):

        # Known (analytical solution)
        #   N(t) = K / ( 1 + (K/N_0 - 1)*exp(-r*t))
        # Known in this function
        #   N_0 = 1, K = N_0 + D = 1 + D, N_tau = ratio*N_0
        #
        # Compute r = max_rate from the above
        #   N_tau = ratio*N_0
        #   N(tau) = N_tau = ratio*N_0 = K / ( 1 + (K/N_0 - 1)*exp(-r*tau))
        #   ratio = (1 + D) / ( 1 + ((1+D)/1 - 1)*exp(-r*tau))
        #   ratio = (1 + D) / ( 1 + D*exp(-r*tau))
        #   1 + D*exp(-r*tau) = (1 + D)/ratio
        #   exp(-r*tau) = (((1 + D)/ratio)-1)/D
        #   r = -log((((1 + D)/ratio)-1)/D)/tau

        # Choosing even smaller D leads to issues with Euler integration (tested empirically)
        D = 10e-7
        max_rate = -np.log((((1 + D) / y_tau_0_ratio) - 1) / D) / tau

        # Known (see _get_ks())
        #   K = N_0*(1+(1/exp(-r*t_infl)))
        # Known in this function
        #   N_0 = 1, K = N_0 + D = 1 + D, r < 0
        #
        # Compute inflection time from the above
        #   1 + D = 1*(1+(1/exp(-r*t_infl)))
        #   D = 1/exp(-r*t_infl)
        #   1/D = exp(-r*t_infl)
        #   -ln(1/D) = r*t_infl
        # The above defined a relationship between r and t_infl for a given D
        t_infl = -np.log(1 / D) / max_rate
        inflection_ratio = t_infl / tau
        dyn_sys = cls(tau, np.ones((n_dims,)), max_rate, inflection_ratio)

        return dyn_sys

    @DynamicalSystem.tau.setter
    def tau(self, new_tau):
        """ Set the time constant.

        @param new_tau: Time constant
        """
        self._tau = new_tau
        self._Ks_cached = None  # Forces recomputing Ks

    @DynamicalSystem.x_init.setter
    def x_init(self, new_x_init):
        """ Set the initial state of the dynamical system.

        @param new_x_init: Initial state of the dynamical system.
        """
        if new_x_init.size != self._dim_x:
            raise ValueError(f"x_init must have size {self._dim_x}")
        self._x_init = new_x_init
        self._Ks_cached = None  # Forces recomputing Ks

    @DynamicalSystem.y_init.setter
    def y_init(self, new_y_init):
        """ Set the initial state of the dynamical system (y part)

        @param new_y_init: Initial state of the dynamical system. (y part)

        Note that for an ExponentialSystem y is equivalent to x.
        """
        self.x_init = new_y_init
        self._Ks_cached = None  # Forces recomputing Ks

    def differential_equation(self, x):
        """ The differential equation which defines the system.

        It relates state values to rates of change of those state values.

        @param x: current state
        @return: xd - rate of change in state
        """
        ks = self._get_ks()
        xd = self._max_rate * x * (1 - (np.divide(x, ks)))
        return xd

    def analytical_solution(self, ts):
        """
         Return analytical solution of the system at certain times.

         @param ts: A vector of times for which to compute the analytical solutions
         @return: (xs, xds) - Sequence of states and their rates of change.
        """

        xs = np.empty([ts.size, self._dim_x])
        xds = np.empty([ts.size, self._dim_x])

        Ks = self._get_ks()  # noqa

        for dd in range(self._dim_x):
            r = self._max_rate[dd] if isinstance(self._max_rate, np.ndarray) else self._max_rate
            exp_rt = np.exp(-r * ts)

            # Auxiliary variables to improve legibility
            K = Ks[dd]  # noqa
            b = (K / self._x_init[dd]) - 1
            xs[:, dd] = K / (1 + b * exp_rt)
            xds[:, dd] = np.multiply((K * r * b) / np.square(1.0 + b * exp_rt), exp_rt)

        return xs, xds

    def _get_ks(self):
        if self._Ks_cached is not None:
            # Cache available; simply return it.
            return self._Ks_cached

        # Variable rename so that it is the same as on the Wikipedia page
        N_0s = self.x_init  # noqa

        # The idea here is that the initial state (called N_0s above), max_rate (r above) and the
        # inflection_ratio are set by the user.
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
        self._Ks_cached = np.empty(N_0s.shape)
        for dd in range(len(N_0s)):
            r = self._max_rate[dd] if isinstance(self._max_rate, np.ndarray) else self._max_rate
            infl_ratio = (
                self._inflection_ratio[dd]
                if isinstance(self._inflection_ratio, np.ndarray)
                else self._inflection_ratio
            )
            t_infl = self.tau * infl_ratio
            self._Ks_cached[dd] = N_0s[dd] * (1.0 + (1.0 / np.exp(-r * t_infl)))

            # If Ks is too close to N_0===initial_state, then the differential equation will always
            # return 0. See differential_equation below
            #   xd = max_rate_*x*(1-(x/Ks))
            # For initial_state this is
            #   xd = max_rate_*initial_state*(1-(initial_state/Ks))
            # If initial_state is very close/equal to Ks we get
            #   xd = max_rate_*Ks*(1-(Ks/Ks))
            #   xd = max_rate_*Ks*(1-1)
            #   xd = max_rate_*Ks*0
            #   xd = 0
            # And integration fails, especially for Euler integration.
            # So we now give a warning if this is likely to happen.
            div = np.divide(N_0s[dd], self._Ks_cached[dd]) - 1.0
            if np.any(np.abs(div) < 10e-9):  # 10e-9 determined empirically
                print(
                    f"In function SigmoidSystem, Ks is too close to N_0s. This may lead to errors "
                    f"during numerical integration. Recommended solution: choose a lower magnitude "
                    f"for the maximum rate of change (currently it is {r}) "
                )

        return self._Ks_cached

    def decouple_parameters(self):
        if np.isscalar(self._max_rate):
            self._max_rate = np.full((self.dim_x,), float(self._max_rate))
        if np.isscalar(self._inflection_ratio):
            self._inflection_ratio = np.full((self.dim_x,), float(self._inflection_ratio))
        self._Ks_cached = None
