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


class TimeSystem(DynamicalSystem):
    def __init__(self, tau, count_down=False):
        """ Initialize a TimeSystem
        
        Args:        
            tau - Time constant
            count_down - Whether timer increases (False) or decreases (True)
        """
        if count_down:
            # Count-down from 1 to 0
            x_init = np.ones((1, 1))
        else:
            # Count-up from 0 to 1
            x_init = np.zeros((1, 1))
        super().__init__(1, tau, x_init)
        self._count_down = count_down

    def differentialEquation(self, x):
        """ The differential equation which defines the system.
        
        It relates state values to rates of change of those state values.
        
        Args: x - current state
        Returns: xd - rate of change in state
        """
        xd = np.zeros([1, 1])
        if self._count_down:
            if x > 0:
                xd[0] = -1.0 / self._tau
        else:
            if x < 1.0:
                xd[0] = 1.0 / self._tau

        return xd

    def analyticalSolution(self, ts):
        """
         Return analytical solution of the system at certain times.
        
         Args: ts - A vector of times for which to compute the analytical solutions 
         Returns: (xs, xds) - Sequence of states and their rates of change.
        """

        T = ts.size

        if self._count_down:
            xs = 1.0 - np.reshape(ts / self.tau, (T, 1))
            xds = -np.ones((T, 1)) / self.tau
            xds[xs < 0.0] = 0.0
            xs[xs < 0.0] = 0.0
        else:
            xs = np.reshape(ts / self.tau, (T, 1))
            xds = np.ones((T, 1)) / self.tau
            xds[xs > 1.0] = 0.0
            xs[xs > 1.0] = 1.0

        return (xs, xds)
