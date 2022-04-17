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


import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Include scripts for plotting
lib_path = os.path.abspath("../../../python/")
sys.path.append(lib_path)

from dynamicalsystems.dynamicalsystems_plotting import *
from dynamicalsystems.ExponentialSystem import ExponentialSystem
from dynamicalsystems.SigmoidSystem import SigmoidSystem
from dynamicalsystems.SpringDamperSystem import SpringDamperSystem
from dynamicalsystems.TimeSystem import TimeSystem


def set_style(lines, label):

    if label == "analytical":
        plt.setp(lines, linestyle="-", linewidth=5, color=(0.8, 0.8, 0.8))
    elif label == "euler":
        plt.setp(lines, linestyle="--", linewidth=2, color=(0.8, 0.0, 0.0))
    elif label == "rungekutta":
        plt.setp(lines, linestyle="--", linewidth=2, color=(0.0, 0.0, 0.0))

    # plt.setp(lines[1],plt.getp(lines[0]))

    plt.setp(lines[0], label=label)


if __name__ == "__main__":

    ###########################################################################
    # Create all systems and add them to a dictionary

    # ExponentialSystem
    tau = 0.6  # Time constant
    x_init = np.array([0.5, 1.0])
    x_attr = np.array([0.8, 0.1])
    alpha = 6.0  # Decay factor
    dyn_systems = {"Exponential": ExponentialSystem(tau, x_init, x_attr, alpha)}

    # TimeSystem
    dyn_systems["Time"] = TimeSystem(tau)

    # TimeSystem (but counting down instead of up)
    count_down = True
    dyn_systems["TimeCountDown"] = TimeSystem(tau, count_down)

    # SigmoidSystem
    max_rate = -10
    inflection_ratio = 0.8
    dyn_systems["Sigmoid"] = SigmoidSystem(tau, x_init, max_rate, inflection_ratio)

    # SpringDamperSystem
    alpha = 12.0
    dyn_systems["SpringDamper"] = SpringDamperSystem(tau, x_init, x_attr, alpha)

    ###########################################################################
    # Start integration of all systems

    # Settings for the integration of the system
    dt = 0.01  # Integration step duration
    integration_duration = 1.5 * tau  # Integrate for longer than the time constant
    n_time_steps = int(np.ceil(integration_duration / dt)) + 1
    # Generate a vector of times, i.e. 0.0, dt, 2*dt, 3*dt .... n_time_steps*dt=integration_duration
    ts = np.linspace(0.0, integration_duration, n_time_steps)

    figure_number = 1
    for name, dyn_system in dyn_systems.items():

        n_plots = 3 if name == "SpringDamper" else 2
        fig = plt.figure(figure_number, figsize=(5 * n_plots, 4))
        figure_number += 1
        axs = [fig.add_subplot(1, n_plots, p + 1) for p in range(n_plots)]

        # Analytical solution
        xs, xds = dyn_system.analyticalSolution(ts)
        lines = dyn_system.plot(ts, xs, xds, axs=axs)
        set_style(lines, "analytical")

        # Euler integration
        xs[0, :], xds[0, :] = dyn_system.integrateStart()
        for ii in range(1, n_time_steps):
            xs[ii, :], xds[ii, :] = dyn_system.integrateStepEuler(dt, xs[ii - 1, :])
        lines = dyn_system.plot(ts, xs, xds, axs=axs)
        set_style(lines, "euler")

        # Runge-kutta integration
        xs[0, :], xds[0, :] = dyn_system.integrateStart()
        for ii in range(1, n_time_steps):
            xs[ii, :], xds[ii, :] = dyn_system.integrateStepRungeKutta(
                dt, xs[ii - 1, :]
            )
        lines = dyn_system.plot(ts, xs, xds, axs=axs)
        set_style(lines, "rungekutta")

        # Runge-kutta integration with different tau
        dyn_system.tau = 1.5 * tau
        xs[0, :], xds[0, :] = dyn_system.integrateStart()
        for ii in range(1, n_time_steps):
            xs[ii, :], xds[ii, :] = dyn_system.integrateStepRungeKutta(
                dt, xs[ii - 1, :]
            )
        lines = dyn_system.plot(ts, xs, xds, axs=axs)
        set_style(lines, "tau")
        dyn_system.tau = tau

        # Runge-kutta integration with a perturbation
        xs[0, :], xds[0, :] = dyn_system.integrateStart()
        for ii in range(1, n_time_steps):
            if ii == int(np.ceil(0.3 * n_time_steps)):
                xs[ii - 1, :] = xs[ii - 1, :] - 0.2
            xs[ii, :], xds[ii, :] = dyn_system.integrateStepRungeKutta(
                dt, xs[ii - 1, :]
            )
        lines = dyn_system.plot(ts, xs, xds, axs=axs)
        set_style(lines, "perturb")

        # Runge-kutta integration with a different attractor
        if name == "Exponential" or name == "SpringDamper":
            dyn_system.y_attr = x_attr - 0.2
            xs[0, :], xds[0, :] = dyn_system.integrateStart()
            for ii in range(1, n_time_steps):
                xs[ii, :], xds[ii, :] = dyn_system.integrateStepRungeKutta(
                    dt, xs[ii - 1, :]
                )
            lines = dyn_system.plot(ts, xs, xds, axs=axs)
            set_style(lines, "attractor")
            dyn_system.y_attr = x_attr

        axs[0].legend()

        fig.suptitle(name)

        # fig.savefig(f'{name}.png')

    plt.show()
