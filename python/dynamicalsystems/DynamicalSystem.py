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

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from abc import ABC, abstractmethod

lib_path = os.path.abspath("../../python/")
sys.path.append(lib_path)


class DynamicalSystem(ABC):
    def __init__(self, order, tau, y_init, n_dims_x=None):
        assert order == 1 or order == 2

        self.dim_y_ = len(y_init)
        self.dim_x_ = n_dims_x * order if n_dims_x else self.dim_y_ * order
        self.tau_ = tau

        self.set_y_init(y_init)

    def get_x_init(self):
        return self.x_init_

    def set_x_init(self, x_init):
        self.x_init_ = np.atleast_1d(x_init)

    def get_y_init(self):
        # if dim_y_==dim_x_, this returns all of x_init
        return self.x_init_[: self.dim_y_]

    def set_y_init(self, y_init):
        # Pad the end with zeros for x = [y 0]
        self.x_init_ = np.zeros(self.dim_x_)
        self.x_init_[: self.dim_y_] = y_init

    @abstractmethod
    def differentialEquation(self, x):
        pass

    def analyticalSolution(self, ts):
        # Default implementation: call differentialEquation
        n_time_steps = ts.size
        xs = np.zeros([n_time_steps, self.dim_x_])
        xds = np.zeros([n_time_steps, self.dim_x_])

        (xs[0, :], xds[0, :]) = self.integrateStart()
        for tt in range(1, n_time_steps):
            dt = ts[tt] - ts[tt - 1]
            (xs[tt, :], xds[tt, :]) = self.integrateStepRungeKutta(dt, xs[tt - 1, :])

        return (xs, xds)

    def integrateStart(self, y_init=None):
        if y_init:
            self.set_y_init(y_init)
        x = self.x_init_
        return (x, self.differentialEquation(x))

    def integrateStep(self, dt, x):
        return self.integrateStepRungeKutta(dt, x)

    def integrateStepEuler(self, dt, x):
        assert dt > 0.0
        assert x.size == self.dim_x_
        xd_updated = self.differentialEquation(x)
        x_updated = x + dt * xd_updated
        return (x_updated, xd_updated)

    def integrateStepRungeKutta(self, dt, x):
        # 4th order Runge-Kutta for a 1st order system
        # http://en.wikipedia.org/wiki/Runge-Kutta_method#The_Runge.E2.80.93Kutta_method

        assert dt > 0.0
        assert x.size == self.dim_x_

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

    def set_tau(self, tau):
        assert tau > 0.0
        self.tau_ = tau

    def plot(self, ts, xs, xds, axs):

        # Prepare tex intepretation
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        if self.dim_x_ == self.dim_y_:

            lines = axs[0].plot(ts, xs)
            axs[0].set_ylabel(r"$x$")

            lines[len(lines) :] = axs[1].plot(ts, xds)
            axs[1].set_ylabel(r"$\dot{x}$")

        else:
            # data has following format: [ y_1..y_D  z_1..z_D   yd_1..yd_D  zd_1..zd_D ]

            ys = xs[:, 0 * self.dim_y_ : 1 * self.dim_y_]
            zs = xs[:, 1 * self.dim_y_ : 2 * self.dim_y_]
            yds = xds[:, 0 * self.dim_y_ : 1 * self.dim_y_]
            zds = xds[:, 1 * self.dim_y_ : 2 * self.dim_y_]

            lines = axs[0].plot(ts, ys)
            axs[0].set_ylabel(r"$y$")

            lines[len(lines) :] = axs[1].plot(ts, yds)
            axs[1].set_ylabel(r"$\dot{y} = z/\tau$")

            lines[len(lines) :] = axs[2].plot(ts, zds / self.tau_)
            axs[2].set_ylabel(r"$\ddot{y} = \dot{z}/\tau$")

        for ax in axs:
            ax.set_xlabel(r"time ($s$)")
            # ax.axis('tight')
            ax.grid()

        return lines
