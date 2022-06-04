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
"""Script for dynamical systems demo."""


import matplotlib.pyplot as plt
import numpy as np

from dmpbbo.dynamicalsystems.ExponentialSystem import ExponentialSystem
from dmpbbo.dynamicalsystems.SigmoidSystem import SigmoidSystem
from dmpbbo.dynamicalsystems.SpringDamperSystem import SpringDamperSystem
from dmpbbo.dynamicalsystems.TimeSystem import TimeSystem


def main():
    """ Main function of the script. """
    ###########################################################################
    # Create all systems and add them to a dictionary

    # ExponentialSystem
    tau = 0.6  # Time constant
    x_init = np.array([0.5, 1.0])
    x_attr = np.array([0.8, 0.1])
    alpha = 6.0  # Decay factor
    dyn_systems = {"Exponential": ExponentialSystem(tau, x_init, x_attr, alpha)}  # noqa

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

    for name, dyn_system in dyn_systems.items():

        # Analytical solution
        xs, xds = dyn_system.analytical_solution(ts)
        lines, axs = dyn_system.plot(ts, xs, xds)
        plt.setp(lines, linestyle="-", linewidth=5, color=(0.8, 0.8, 0.8))
        plt.setp(lines[0], label="analytical")

        # Euler integration
        xs[0, :], xds[0, :] = dyn_system.integrate_start()
        for ii in range(1, n_time_steps):
            xs[ii, :], xds[ii, :] = dyn_system.integrate_step_euler(dt, xs[ii - 1, :])
        lines, _ = dyn_system.plot(ts, xs, xds, axs=axs)
        plt.setp(lines, linestyle="--", linewidth=2, color=(0.8, 0.0, 0.0))
        plt.setp(lines[0], label="Euler")

        # Runge-kutta integration
        xs[0, :], xds[0, :] = dyn_system.integrate_start()
        for ii in range(1, n_time_steps):
            xs[ii, :], xds[ii, :] = dyn_system.integrate_step_runge_kutta(dt, xs[ii - 1, :])
        lines, _ = dyn_system.plot(ts, xs, xds, axs=axs)
        plt.setp(lines, linestyle="--", linewidth=2, color=(0.2, 0.2, 0.2))
        plt.setp(lines[0], label="Runge-Kutta")

        # Runge-kutta integration with different tau
        dyn_system.tau = 1.5 * tau
        xs[0, :], xds[0, :] = dyn_system.integrate_start()
        for ii in range(1, n_time_steps):
            xs[ii, :], xds[ii, :] = dyn_system.integrate_step_runge_kutta(dt, xs[ii - 1, :])
        lines, _ = dyn_system.plot(ts, xs, xds, axs=axs)
        plt.setp(lines, linestyle="-", linewidth=1, color=(0.2, 0.8, 0.2))
        plt.setp(lines[0], label="tau")
        dyn_system.tau = tau

        # Runge-kutta integration with a perturbation
        xs[0, :], xds[0, :] = dyn_system.integrate_start()
        for ii in range(1, n_time_steps):
            if ii == int(np.ceil(0.3 * n_time_steps)):
                xs[ii - 1, :] = xs[ii - 1, :] - 0.2
            xs[ii, :], xds[ii, :] = dyn_system.integrate_step_runge_kutta(dt, xs[ii - 1, :])
        lines, _ = dyn_system.plot(ts, xs, xds, axs=axs)
        plt.setp(lines, linestyle="-", linewidth=1, color=(0.2, 0.2, 0.8))
        plt.setp(lines[0], label="perturbation")

        # Runge-kutta integration with a different attractor
        if name == "Exponential" or name == "SpringDamper":
            dyn_system.y_attr = x_attr - 0.2
            xs[0, :], xds[0, :] = dyn_system.integrate_start()
            for ii in range(1, n_time_steps):
                xs[ii, :], xds[ii, :] = dyn_system.integrate_step_runge_kutta(dt, xs[ii - 1, :])
            lines, _ = dyn_system.plot(ts, xs, xds, axs=axs)
            plt.setp(lines, linestyle="-", linewidth=1, color=(0.8, 0.2, 0.8))
            plt.setp(lines[0], label="attractor")
            dyn_system.y_attr = x_attr

        axs[0].legend()

        plt.gcf().suptitle(name)

        # fig.savefig(f'{name}.png')

    plt.show()


if __name__ == "__main__":
    main()
