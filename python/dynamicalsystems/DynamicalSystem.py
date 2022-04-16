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
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

class DynamicalSystem(ABC):
    def __init__(self, order, tau, y_init, n_dims_x=None):
        if order<1 or order>2:
            raise ValueError("order should be 1 or 2")

        # These are set once, and are fixed 
        self._dim_y = len(y_init)
        self._dim_x = n_dims_x * order if n_dims_x else self._dim_y * order
        
        self.tau = tau
        self.y_init = y_init

    @property
    def tau(self):
        return self._tau
   
    @tau.setter
    def tau(self, new_tau):
        if new_tau<=0.0:
            raise ValueError("tau should be larger than 0.0")
        self._tau = new_tau

    @property
    def x_init(self):
        return self._x_init

    @x_init.setter
    def x_init(self, x):
        if x.size!=self._dim_x:
            raise ValueError("x_init must have size "+self._dim_x)
        self._x_init = np.atleast_1d(x)

    @property
    def y_init(self):
        # if _dim_y==_dim_x, this returns all of x_init
        return self._x_init[: self._dim_y]

    @y_init.setter
    def y_init(self, y):
        if y.size!=self._dim_y:
            raise ValueError("y_init must have size "+self._dim_y)
        # Pad the end with zeros for x = [y 0]
        self._x_init = np.zeros(self._dim_x)
        self._x_init[: self._dim_y] = y

    @abstractmethod
    def differentialEquation(self, x):
        pass

    def analyticalSolution(self, ts):
        # Default implementation: call differentialEquation
        n_time_steps = ts.size
        xs = np.zeros([n_time_steps, self._dim_x])
        xds = np.zeros([n_time_steps, self._dim_x])

        (xs[0, :], xds[0, :]) = self.integrateStart()
        for tt in range(1, n_time_steps):
            dt = ts[tt] - ts[tt - 1]
            (xs[tt, :], xds[tt, :]) = self.integrateStepRungeKutta(dt, xs[tt - 1, :])

        return (xs, xds)

    def integrateStart(self, y_init=None):
        if y_init:
            self.y_init = y_init
        x = self._x_init
        return (x, self.differentialEquation(x))

    def integrateStep(self, dt, x):
        return self.integrateStepRungeKutta(dt, x)

    def integrateStepEuler(self, dt, x):
        assert dt > 0.0
        assert x.size == self._dim_x
        xd_updated = self.differentialEquation(x)
        x_updated = x + dt * xd_updated
        return (x_updated, xd_updated)

    def integrateStepRungeKutta(self, dt, x):
        # 4th order Runge-Kutta for a 1st order system
        # http://en.wikipedia.org/wiki/Runge-Kutta_method#The_Runge.E2.80.93Kutta_method

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
        return (x_updated, xd_updated)

    def plot(self, ts, xs, xds, axs):

        # Prepare tex intepretation
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        if self._dim_x == self._dim_y:

            lines = axs[0].plot(ts, xs)
            axs[0].set_ylabel(r"$x$")

            lines[len(lines) :] = axs[1].plot(ts, xds)
            axs[1].set_ylabel(r"$\dot{x}$")

        else:
            # data has following format: [ y_1..y_D  z_1..z_D   yd_1..yd_D  zd_1..zd_D ]

            ys = xs[:, 0 * self._dim_y : 1 * self._dim_y]
            zs = xs[:, 1 * self._dim_y : 2 * self._dim_y]
            yds = xds[:, 0 * self._dim_y : 1 * self._dim_y]
            zds = xds[:, 1 * self._dim_y : 2 * self._dim_y]

            lines = axs[0].plot(ts, ys)
            axs[0].set_ylabel(r"$y$")

            lines[len(lines) :] = axs[1].plot(ts, yds)
            axs[1].set_ylabel(r"$\dot{y} = z/\tau$")

            lines[len(lines) :] = axs[2].plot(ts, zds / self._tau)
            axs[2].set_ylabel(r"$\ddot{y} = \dot{z}/\tau$")

        for ax in axs:
            ax.set_xlabel(r"time ($s$)")
            # ax.axis('tight')
            ax.grid()

        return lines
