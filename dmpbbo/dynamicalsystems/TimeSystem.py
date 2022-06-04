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
""" Module for the TimeSystem class. """

import numpy as np

from dmpbbo.dynamicalsystems.DynamicalSystem import DynamicalSystem


class TimeSystem(DynamicalSystem):
    """ A dynamical system with constant velocity, representing the linear passage of time.
    """

    def __init__(self, tau, count_down=False):
        """ Initialize a TimeSystem

        @param tau: Time constant
        @param count_down: Whether timer increases (False) or decreases (True)
        """
        # Count-down from 1 to 0 or count-up from 0 to 1
        self._count_down = count_down
        x_init = 1.0 if self._count_down else 0.0
        super().__init__(1, tau, np.array(x_init))

    @property
    def count_down(self):
        """ Get whether this TimeSystem counts down or up.

        @return: true if counting down, false if counting up
        """
        return self._count_down

    @count_down.setter
    def count_down(self, new_count_down):
        self._count_down = new_count_down
        x_init = 1.0 if self._count_down else 0.0
        self.x_init = np.array(x_init)

    def differential_equation(self, x):
        """ The differential equation which defines the system.

        It relates state values to rates of change of those state values.

        @param x: current state
        @return: xd - rate of change in state
        """
        xd = np.zeros([1, 1])
        if self._count_down:
            if x > 0:
                xd[0] = -1.0 / self._tau
        else:
            if x < 1.0:
                xd[0] = 1.0 / self._tau

        return xd

    def analytical_solution(self, ts):
        """
         Return analytical solution of the system at certain times.

         @param ts: A vector of times for which to compute the analytical solutions
         @return: (xs, xds) - Sequence of states and their rates of change.
        """

        n_time_steps = ts.size

        if self._count_down:
            xs = 1.0 - np.reshape(ts / self.tau, (n_time_steps, 1))
            xds = -np.ones((n_time_steps, 1)) / self.tau
            xds[xs < 0.0] = 0.0
            xs[xs < 0.0] = 0.0
        else:
            xs = np.reshape(ts / self.tau, (n_time_steps, 1))
            xds = np.ones((n_time_steps, 1)) / self.tau
            xds[xs > 1.0] = 0.0
            xs[xs > 1.0] = 1.0

        return xs, xds
