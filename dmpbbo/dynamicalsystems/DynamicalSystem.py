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

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class DynamicalSystem(ABC):
    def __init__(self, order, tau, y_init, n_dims_x=None):
        """ Initialize a first or second order dynamical system.
        
        Args:
            order    - Order of the system (1 or 2)
            tau      - Time constant
            y_init   - Initial state
            n_dims_x - Dimensionality of the state (which may differ from the size  of y_init) 
        """
        if order < 1 or order > 2:
            raise ValueError("order should be 1 or 2")

        self.tau = tau

        y_init = np.atleast_1d(y_init)  # In case it is a scalar

        # These are set once, and are fixed
        self._dim_y = y_init.size
        self._dim_x = n_dims_x * order if n_dims_x else self._dim_y * order

        self._x_init = np.zeros(self._dim_x)
        self._x_init[: self._dim_y] = y_init

    @property  # Needs to be a property, so that subclasses can override setter method
    def tau(self):
        """ Get the time constant.
         
         Returns: Time constant
        """
        return self._tau

    @tau.setter
    def tau(self, new_tau):
        """ Set the time constant.
         
         Args:
            new_tau - Time constant
        """
        self._tau = new_tau

    @property
    def x_init(self):
        """ Get the initial state of the dynamical system. 
        """
        return self._x_init

    @x_init.setter
    def x_init(self, new_x_init):
        """ Set the initial state of the dynamical system.
        
         Args:
            new_x_init Initial state of the dynamical system.
        """
        if new_x_init.size != self._dim_x:
            raise ValueError("x_init must have size " + self._dim_x)
        self._x_init = np.atleast_1d(new_x_init)

    @property
    def y_init(self):
        """
         Get the y part of the initial state of the dynamical system.
        
        Returns: Initial state of the dynamical system.
        """
        # if _dim_y==_dim_x, this returns all of x_init
        return self._x_init[: self._dim_y]

    @y_init.setter
    def y_init(self, y_init_new):
        """ Set the y part of the initial state of the dynamical system.
        
        Args:
            y_init_new Initial state of the dynamical system.
        """
        if y_init_new.size != self._dim_y:
            raise ValueError("y_init_new must have size " + self._dim_y)
        # Pad the end with zeros for x = [y 0]
        x_init_new = np.zeros(self._dim_x)
        x_init_new[: self._dim_y] = y_init_new
        self.x_init = x_init_new

    @abstractmethod
    def differentialEquation(self, x):
        """ The differential equation which defines the system.
        
        It relates state values to rates of change of those state values.
        
        Args: x - current state
        Returns: xd - rate of change in state
        """
        pass

    @abstractmethod
    def analyticalSolution(self, ts):
        """
         Return analytical solution of the system at certain times.
        
         Args: ts - A vector of times for which to compute the analytical solutions 
         Returns: (xs, xds) - Sequence of states and their rates of change.
        """
        pass

    def integrateStart(self, y_init=None):
        """ Start integrating the system with a new initial state.
        
        Args:
            y_init - The initial state vector (y part)
        Returns:
            x, xd - The first vector of state variables and their rates of change
        """
        if y_init:
            self.y_init = y_init
        x = self._x_init
        return x, self.differentialEquation(x)

    def integrateStep(self, dt, x):
        """ Integrate the system one time step.
        
        Args:
            dt - Duration of the time step
            x - Current state
            
        Returns:    
            (x_updated, xd_updated) - Updated state and its rate of change, dt time later.
        """
        return self.integrateStepRungeKutta(dt, x)

    def integrateStepEuler(self, dt, x):
        """ Integrate the system one time step using Euler integration. 
        
        Args:
            dt - Duration of the time step
            x - Current state
            
        Returns:    
            (x_updated, xd_updated) - Updated state and its rate of change, dt time later.
        """
        assert dt > 0.0
        assert x.size == self._dim_x
        xd_updated = self.differentialEquation(x)
        x_updated = x + dt * xd_updated
        return x_updated, xd_updated

    def integrateStepRungeKutta(self, dt, x):
        """Integrate the system one time step using 4th order Runge-Kutta integration. 
        
        See http://en.wikipedia.org/wiki/Runge-Kutta_method#The_Runge.E2.80.93Kutta_method
        
        
        Args:
            dt - Duration of the time step
            x - Current state
            
        Returns:    
            (x_updated, xd_updated) - Updated state and its rate of change, dt time later.
        """

        assert dt > 0.0
        assert x.size == self._dim_x

        k1 = self.differentialEquation(x)
        input_k2 = x + dt * 0.5 * k1
        k2 = self.differentialEquation(input_k2)
        input_k3 = x + dt * 0.5 * k2
        k3 = self.differentialEquation(input_k3)
        input_k4 = x + dt * k3
        k4 = self.differentialEquation(input_k4)

        x_updated = x + dt * (k1 + 2.0 * (k2 + k3) + k4) / 6.0
        xd_updated = self.differentialEquation(x_updated)
        return x_updated, xd_updated

    def plot(self, ts, xs, xds, **kwargs):
        kwargs["dim_y"] = self._dim_y
        return DynamicalSystem.plotStatic(self._tau, ts, xs, xds, **kwargs)

    @staticmethod
    def plotStatic(tau, ts, xs, xds, **kwargs):
        """Plot the output of the integration of a dynamical system.
        
        Args:
            ts - Times at which the state was determined (size: n_time_steps)
            xs, xds - System states and its rates of change (shape: n_time_steps X n_dim_x)
            
        Kwargs:
            dim_y - Dimensionality of y part of state, i.e. x = [y z]. Default: dim_y = dim_x
            axs - Axes on which the plot the output
            fig - Figure on which to plot the output
        """

        dim_x = xs.shape[1]
        dim_y = kwargs.pop("dim_y", dim_x)

        if "axs" in kwargs:
            axs = kwargs["axs"]
        else:
            # 2 => plot x and xd, 3 => plot y, yd and ydd=zd/tau
            n_plots = 2 if dim_x == dim_y else 3
            fig = kwargs.get("fig") or plt.figure(figsize=(5 * n_plots, 4))
            axs = [fig.add_subplot(1, n_plots, p + 1) for p in range(n_plots)]

        # Prepare tex interpretation
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        if dim_x == dim_y:

            lines = axs[0].plot(ts, xs)
            axs[0].set_ylabel(r"$x$")

            if len(axs) > 1:
                lines[len(lines) :] = axs[1].plot(ts, xds)
                axs[1].set_ylabel(r"$\dot{x}$")

        else:
            # data has following format: [ y_1..y_D  z_1..z_D   yd_1..yd_D  zd_1..zd_D ]

            ys = xs[:, 0 * dim_y : 1 * dim_y]
            # zs = xs[:, 1 * dim_y : 2 * dim_y]
            yds = xds[:, 0 * dim_y : 1 * dim_y]
            zds = xds[:, 1 * dim_y : 2 * dim_y]

            lines = axs[0].plot(ts, ys)
            axs[0].set_ylabel(r"$y$")

            if len(axs) > 1:
                lines[len(lines) :] = axs[1].plot(ts, yds)
                axs[1].set_ylabel(r"$\dot{y} = z/\tau$")

            if len(axs) > 2:
                lines[len(lines) :] = axs[2].plot(ts, zds / tau)
                axs[2].set_ylabel(r"$\ddot{y} = \dot{z}/\tau$")

        for ax in axs:
            ax.set_xlabel(r"time ($s$)")
            # ax.axis('tight')
            ax.grid()

        return lines, axs
