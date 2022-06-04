# This file is part of DmpBbo, a set of libraries and programs for the
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2022 Freek Stulp
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


import argparse
import os
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np

import dmpbbo.json_for_cpp as jc
from dmpbbo.dynamicalsystems.ExponentialSystem import ExponentialSystem
from dmpbbo.dynamicalsystems.SigmoidSystem import SigmoidSystem
from dmpbbo.dynamicalsystems.SpringDamperSystem import SpringDamperSystem
from dmpbbo.dynamicalsystems.TimeSystem import TimeSystem
from tests.integration.execute_binary import execute_binary


def plot_comparison(ts, xs, xds, xs_cpp, xds_cpp, fig):
    axs = [fig.add_subplot(2, 2, p + 1) for p in range(4)]

    # plt.rc("text", usetex=True)
    # plt.rc("font", family="serif")

    h_cpp = []
    h_pyt = []
    h_diff = []

    h_pyt.extend(axs[0].plot(ts, xs, label="Python"))
    h_cpp.extend(axs[0].plot(ts, xs_cpp, label="C++"))
    axs[0].set_ylabel("x")

    h_pyt.extend(axs[1].plot(ts, xds, label="Python"))
    h_cpp.extend(axs[1].plot(ts, xds_cpp, label="C++"))
    axs[1].set_ylabel("dx")

    # Reshape needed when xs_cpp has shape (T,)
    h_diff.extend(axs[2].plot(ts, xs - np.reshape(xs_cpp, xs.shape), label="diff"))
    axs[2].set_ylabel("diff x")

    h_diff.extend(axs[3].plot(ts, xds - np.reshape(xds_cpp, xds.shape), label="diff"))
    axs[3].set_ylabel("diff xd")

    plt.setp(h_pyt, linestyle="-", linewidth=4, color=(0.8, 0.8, 0.8))
    plt.setp(h_cpp, linestyle="--", linewidth=2, color=(0.2, 0.2, 0.8))
    plt.setp(h_diff, linestyle="-", linewidth=1, color=(0.8, 0.2, 0.2))

    for ax in axs:
        ax.set_xlabel("$t$")
        ax.legend()

    pass


def test_dynamical_systems(tmp_path):
    main(tmp_path)


def main(directory, **kwargs):
    show = kwargs.get("show", False)
    save = kwargs.get("save", False)
    verbose = kwargs.get("verbose", False)

    directory.mkdir(parents=True, exist_ok=True)

    ###########################################################################
    # Create all systems and add them to a dictionary

    # ExponentialSystem
    tau = 0.6  # Time constant
    x_init_2d = np.array([0.5, 1.0])
    x_attr_2d = np.array([0.8, 0.1])
    x_init_1d = np.array([0.5])
    x_attr_1d = np.array([0.8])

    alpha = 6.0
    dyn_systems = {}  # noqa
    dyn_systems["ExponentialSystem_1D"] = ExponentialSystem(tau, x_init_1d, x_attr_1d, alpha)
    dyn_systems["ExponentialSystem_2D"] = ExponentialSystem(tau, x_init_2d, x_attr_2d, alpha)

    # SigmoidSystem
    max_rate = -10
    inflection_ratio = 0.8
    dyn_systems["SigmoidSystem_1D"] = SigmoidSystem(tau, x_init_1d, max_rate, inflection_ratio)
    dyn_systems["SigmoidSystem_2D"] = SigmoidSystem(tau, x_init_2d, max_rate, inflection_ratio)

    # SpringDamperSystem
    alpha = 12.0
    dyn_systems["SpringDamperSystem_1D"] = SpringDamperSystem(tau, x_init_1d, x_attr_1d, alpha)
    dyn_systems["SpringDamperSystem_2D"] = SpringDamperSystem(tau, x_init_2d, x_attr_2d, alpha)

    # TimeSystem
    dyn_systems["TimeSystem"] = TimeSystem(tau)

    # TimeSystem (but counting down instead of up)
    count_down = True
    dyn_systems["TimeCountDownSystem"] = TimeSystem(tau, count_down)

    ###########################################################################
    # Start integration of all systems

    # Settings for the integration of the system
    dt = 0.01  # Integration step duration
    integration_duration = 1.25 * tau  # Integrate for longer than the time constant
    n_time_steps = int(np.ceil(integration_duration / dt)) + 1
    # Generate a vector of times, i.e. 0.0, dt, 2*dt, 3*dt .... n_time_steps*dt=integration_duration
    ts = np.linspace(0.0, integration_duration, n_time_steps)
    # https://youtrack.jetbrains.com/issue/PY-35025
    np.savetxt(os.path.join(directory, "ts.txt"), ts)  # noqa

    for name in dyn_systems.keys():

        dyn_system = dyn_systems[name]

        # Save the dynamical system to a json file
        jc.savejson(Path(directory, f"{name}.json"), dyn_system)
        jc.savejson_for_cpp(Path(directory, f"{name}_for_cpp.json"), dyn_system)

        # Call the binary, which does analytical_solution and integration in C++
        exec_name = "testDynamicalSystems"
        arguments = f"{directory} {name}"
        execute_binary(exec_name, arguments)

        if verbose:
            print("===============")
            print("Python Analytical solution")
        xs, xds = dyn_system.analytical_solution(ts)
        xs_cpp = np.loadtxt(os.path.join(directory, "xs_analytical.txt"))
        xds_cpp = np.loadtxt(os.path.join(directory, "xds_analytical.txt"))
        max_diff = np.max(np.abs(xs - np.reshape(xs_cpp, xs.shape)))
        if verbose:
            print(f"    max_diff = {max_diff}  ({name}System, Analytical)")
        assert max_diff < 10e-7
        if save or show:
            fig1 = plt.figure(figsize=(10, 10))
            plot_comparison(ts, xs, xds, xs_cpp, xds_cpp, fig1)
            fig1.suptitle(f"{name}System - Analytical")

        if verbose:
            print("===============")
            print("Python Integrating with Euler")
        xs[0, :], xds[0, :] = dyn_system.integrate_start()
        for ii in range(1, n_time_steps):
            xs[ii, :], xds[ii, :] = dyn_system.integrate_step_euler(dt, xs[ii - 1, :])
        xs_cpp = np.loadtxt(os.path.join(directory, "xs_euler.txt"))
        xds_cpp = np.loadtxt(os.path.join(directory, "xds_euler.txt"))
        max_diff = np.max(np.abs(xs - np.reshape(xs_cpp, xs.shape)))
        if verbose:
            print(f"    max_diff = {max_diff}  ({name}System, Euler)")
        assert max_diff < 10e-7
        if save or show:
            fig2 = plt.figure(figsize=(10, 10))
            plot_comparison(ts, xs, xds, xs_cpp, xds_cpp, fig2)
            fig2.suptitle(f"{name}System - Euler")

        if verbose:
            print("===============")
            print("Python Integrating with Runge-Kutta")
        xs[0, :], xds[0, :] = dyn_system.integrate_start()
        for ii in range(1, n_time_steps):
            xs[ii, :], xds[ii, :] = dyn_system.integrate_step_runge_kutta(dt, xs[ii - 1, :])
        xs_cpp = np.loadtxt(os.path.join(directory, "xs_rungekutta.txt"))
        xds_cpp = np.loadtxt(os.path.join(directory, "xds_rungekutta.txt"))
        max_diff = np.max(np.abs(xs - np.reshape(xs_cpp, xs.shape)))
        if verbose:
            print(f"    max_diff = {max_diff}  ({name}System, Runge-Kutta)")
        assert max_diff < 10e-7
        if save or show:
            fig3 = plt.figure(figsize=(10, 10))
            plot_comparison(ts, xs, xds, xs_cpp, xds_cpp, fig3)
            fig3.suptitle(f"{name}System - Runge-Kutta")

        if save:
            fig1.savefig(Path(directory, f"{name}System_analytical.png"))  # noqa
            fig2.savefig(Path(directory, f"{name}System_euler.png"))  # noqa
            fig2.savefig(Path(directory, f"{name}System_rungekutta.png"))

    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--show", action="store_true", help="show plots")
    parser.add_argument("--save", action="store_true", help="save plots")
    # parser.add_argument("--verbose", action="store_true", help="print output")
    parser.add_argument(
        "--directory",
        help="directory to write results to",
        default=Path(tempfile.gettempdir(), "dmpbbo", "test_dynamical_systems_data"),
    )
    args = parser.parse_args()

    main(Path(args.directory), show=True, save=args.save, verbose=True)
