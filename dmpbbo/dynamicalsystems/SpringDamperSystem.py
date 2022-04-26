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


from dmpbbo.dynamicalsystems.DynamicalSystem import DynamicalSystem


class SpringDamperSystem(DynamicalSystem):
    def __init__(
        self,
        tau,
        y_init,
        y_attr,
        damping_coefficient,
        spring_constant="CRITICALLY_DAMPED",
        mass=1.0,
    ):
        """ Initialize a SpringDamperSystem.
        
        Args:
            tau    - Time constant
            y_init - Initial state (y part, i.e. x = [y z])
            y_attr - Attractor state (y part, i.e. x = [y z])
            spring_constant - Spring constant. Can be set to "CRITICALLY_DAMPED" (str)
            damping_coefficient - Damping coefficient
            mass - Mass
        """
        super().__init__(2, tau, y_init)
        self._y_attr = y_attr
        self._damping_coefficient = damping_coefficient
        self._mass = mass
        if spring_constant == "CRITICALLY_DAMPED":
            self._spring_constant = damping_coefficient * damping_coefficient / 4
        else:
            self._spring_constant = spring_constant

    @property
    def y_attr(self):
        """ Return the y part of the attractor state, where x = [y z]
        """
        return _y_attr

    @y_attr.setter
    def y_attr(self, new_y_attr):
        """ Set the y part of the attractor state, where x = [y z]
        """
        if new_y_attr.size != self._dim_y:
            raise ValueError("y_attr must have size " + self._dim_y)
        self._y_attr = np.atleast_1d(new_y_attr)

    def differentialEquation(self, x):
        """ The differential equation which defines the system.
        
        It relates state values to rates of change of those state values.
        
        Args: x - current state
        Returns: xd - rate of change in state
        """

        # Spring-damper system was originally 2nd order, i.e. with [x xd xdd]
        # After rewriting it as a 1st order system it becomes [y z yd zd], with yd = z;
        # Get 'y' and 'z' parts of the state in 'x'
        y = x[0 : self._dim_y]
        z = x[self._dim_y :]

        # Compute yd and zd
        # See  http://en.wikipedia.org/wiki/Damped_spring-mass_system#Example:mass_.E2.80.93spring.E2.80.93damper
        # and equation 2.1 of http://www-clmc.usc.edu/publications/I/ijspeert-NC2013.pdf
        yd = z / self._tau

        zd = (
            -self._spring_constant * (y - self._y_attr) - self._damping_coefficient * z
        ) / (self._mass * self._tau)

        xd = np.concatenate((yd, zd))

        return xd

    def analyticalSolution(self, ts):
        """
         Return analytical solution of the system at certain times.
        
         Args: ts - A vector of times for which to compute the analytical solutions 
         Returns: (xs, xds) - Sequence of states and their rates of change.
        """

        n_time_steps = ts.size
        xs = np.zeros((n_time_steps, self._dim_x))
        xds = np.zeros((n_time_steps, self._dim_x))

        # Closed form solution to 2nd order canonical system
        # This system behaves like a critically damped spring-damper system
        # http://en.wikipedia.org/wiki/Damped_spring-mass_system
        omega_0 = (
            np.sqrt(self._spring_constant / self._mass) / self._tau
        )  # natural frequency
        zeta = self._damping_coefficient / (
            2 * np.sqrt(self._mass * self._spring_constant)
        )  # damping ratio
        if zeta != 1.0:
            print(
                f"WARNING: Spring-damper system is not critically damped, zeta={zeta}"
            )

        for i_dim in range(self._dim_y):
            y0 = self._x_init[i_dim] - self._y_attr[i_dim]
            yd0 = self._x_init[self._dim_y + i_dim]

            A = y0
            B = yd0 + omega_0 * y0

            # Closed form solutions
            # See http://en.wikipedia.org/wiki/Damped_spring-mass_system
            exp_term = -omega_0 * ts
            exp_term = np.exp(exp_term)

            Y = 0 * self._dim_y + i_dim
            Z = 1 * self._dim_y + i_dim

            ABts = A + B * ts

            xs[:, Y] = self._y_attr[i_dim] + ABts * exp_term  # .array()

            # Derivative of the above (use product rule: (f*g)' = f'*g + f*g'
            xds[:, Y] = ((B - omega_0 * ABts)) * exp_term  # .array()

            # Derivative of the above (again use product    rule: (f*g)' = f'*g + f*g'
            ydds = (-omega_0 * (2 * B - omega_0 * ABts)) * exp_term

            # This is how to compute the 'z' terms from the 'y' terms
            xs[:, Z] = xds[:, Y] * self._tau
            xds[:, Z] = ydds * self._tau

        return (xs, xds)
