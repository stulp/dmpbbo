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

from dmpbbo.dynamicalsystems.DynamicalSystem import DynamicalSystem  #


class SigmoidSystem(DynamicalSystem):
    def __init__(self, tau, x_init, max_rate, inflection_ratio):
        """ Initialize a SigmoidSystem.
        
        Args:
            tau    - Time constant
            x_init - Initial state
            max_rate - Maximum rate of change
            inflection_ratio - Time at which maximum rate of change is 
                               achieved, i.e. at inflection_ratio * tau
        """
        super().__init__(1, tau, x_init)
        self._tau = tau  # To avoid warnings (is already set by super.init above)
        self._max_rate = max_rate
        self._inflection_ratio = inflection_ratio
        self._Ks_cached = None

    @DynamicalSystem.tau.setter
    def tau(self, new_tau):
        """ Set the time constant.
         
         Args:
            new_tau - Time constant
        """
        self._tau = new_tau
        self._Ks_cached = None  # Forces recomputing Ks

    @DynamicalSystem.x_init.setter
    def x_init(self, new_x_init):
        """ Set the initial state of the dynamical system.
        
         Args:
            new_x_init Initial state of the dynamical system.
        """
        if new_x_init.size != self._dim_x:
            raise ValueError("x_init must have size " + self._dim_x)
        self._x_init = new_x_init
        self._Ks_cached = None  # Forces recomputing Ks

    @DynamicalSystem.y_init.setter
    def y_init(self, new_y_init):
        """ Set the initial state of the dynamical system (y part)
     
         Args:
            new_y_init Initial state of the dynamical system. (y part)
     
        Note that for an ExponentialSystem y is equivalent to x.
        """
        self.x_init = new_y_init
        self._Ks_cached = None  # Forces recomputing Ks

    def differentialEquation(self, x):
        """ The differential equation which defines the system.
        
        It relates state values to rates of change of those state values.
        
        Args: x - current state
        Returns: xd - rate of change in state
        """
        Ks = self._getKs()
        xd = self._max_rate * x * (1 - (np.divide(x, Ks)))
        return xd

    def analyticalSolution(self, ts):
        """
         Return analytical solution of the system at certain times.
        
         Args: ts - A vector of times for which to compute the analytical solutions 
         Returns: (xs, xds) - Sequence of states and their rates of change.
        """

        # Auxiliary variables to improve legibility
        r = self._max_rate
        exp_rt = np.exp(-r * ts)

        xs = np.empty([ts.size, self._dim_x])
        xds = np.empty([ts.size, self._dim_x])

        Ks = self._getKs()

        for dd in range(self._dim_x):
            # Auxiliary variables to improve legibility
            K = Ks[dd]
            b = (K / self._x_init[dd]) - 1
            xs[:, dd] = K / (1 + b * exp_rt)
            xds[:, dd] = np.multiply((K * r * b) / np.square(1.0 + b * exp_rt), exp_rt)

        return xs, xds

    def _getKs(self):
        if self._Ks_cached is not None:
            # Cache available; simply return it.
            return self._Ks_cached

        # Variable rename so that it is the same as on the Wikipedia page
        N_0s = self.x_init
        r = self._max_rate
        t_infl = self.tau * self._inflection_ratio

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
                f"In function SigmoidSystem, Ks is too close to N_0s. This may lead to errors during numerical "
                f"integration. Recommended solution: choose a lower magnitude for the maximum rate of change ("
                f"currently it is {r}) "
            )

        return self._Ks_cached
