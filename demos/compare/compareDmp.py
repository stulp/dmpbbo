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


import os
import subprocess
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import dmpbbo.DmpBboJSONEncoder as dj
from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import (
    FunctionApproximatorRBFN,
)


def executeBinary(executable_name, arguments, print_command=False):

    if not os.path.isfile(executable_name):
        raise ValueError(
            f"Executable '{executable_name}' does not exist. Please call 'make install' in the build directory first."
        )

    command = f"{executable_name} {arguments}"
    if print_command:
        print(command)

    subprocess.call(command, shell=True)


def plotComparison(ts, xs, xds, xs_cpp, xds_cpp, fig):
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


if __name__ == "__main__":

    directory = "/tmp/compareDmp/"
    os.makedirs(directory, exist_ok=True)

    ################################
    # Read trajectory and train DMP with it.
    trajectory_file = "trajectory.txt"
    print(f"Reading trajectory from: {trajectory_file}\n")
    traj = Trajectory.loadtxt(trajectory_file)
    n_dims = traj.dim

    n_bfs = 10
    function_apps = [FunctionApproximatorRBFN(n_bfs, 0.7) for i_dim in range(n_dims)]
    dmp = Dmp.from_traj(traj, function_apps, "KULVICIUS_2012_JOINING")

    ################
    # Analytical solution to compute difference
    print("===============\nPython Analytical solution")

    ts = traj.ts
    xs_ana, xds_ana, forcing_terms_ana, fa_outputs_ana = dmp.analyticalSolution(ts)

    ################################################
    # Numerically integrate the DMP

    print("===============\nPython Numerical integration")
    n_time_steps = len(ts)
    xs_step = np.zeros([n_time_steps, dmp._dim_x])
    xds_step = np.zeros([n_time_steps, dmp._dim_x])

    (x, xd) = dmp.integrateStart()
    xs_step[0, :] = x
    xds_step[0, :] = xd
    for tt in range(1, n_time_steps):
        dt = ts[tt] - ts[tt - 1]
        xs_step[tt, :], xds_step[tt, :] = dmp.integrateStepRungeKutta(
            dt, xs_step[tt - 1, :]
        )

    ################################################
    # Call the binary, which does analyticalSolution and numerical integration in C++

    # Save the dynamical system to a json file
    filename_json = Path(directory, "dmp.json")
    dj.savejson(filename_json, dmp, save_for_cpp_also=True)
    np.savetxt(Path(directory, "ts.txt"), ts)

    exec_name = "../../build_dir_realtime/demos/compare/compareDmp"
    executeBinary(exec_name, f"{directory} dmp", True)

    print("===============\nPython reading output from C++")
    d = directory
    xs_ana_cpp = np.loadtxt(Path(d, "xs_ana.txt"))
    xds_ana_cpp = np.loadtxt(Path(d, "xds_ana.txt"))
    forcing_terms_ana_cpp = np.loadtxt(Path(d, "forcing_terms_ana.txt"))
    fa_outputs_ana_cpp = np.loadtxt(Path(d, "fa_outputs_ana.txt"))
    xs_step_cpp = np.loadtxt(Path(d, "xs_step.txt"))
    xds_step_cpp = np.loadtxt(Path(d, "xds_step.txt"))

    # Plotting

    print("===============\nPython Plotting")
    save_png = True

    h_pyt, axs1 = Dmp.plotStatic(
        dmp.tau,
        ts,
        xs_ana,
        xds_ana,
        forcing_terms=forcing_terms_ana,
        fa_outputs=fa_outputs_ana,
    )
    h_cpp, _ = Dmp.plotStatic(
        dmp.tau,
        ts,
        xs_ana_cpp,
        xds_ana_cpp,
        axs=axs1,
        forcing_terms=forcing_terms_ana_cpp,
        fa_outputs=fa_outputs_ana_cpp,
    )
    plt.setp(h_pyt, linestyle="-", linewidth=4, color=(0.8, 0.8, 0.8))
    plt.setp(h_cpp, linestyle="--", linewidth=2, color=(0.2, 0.2, 0.8))
    plt.gcf().suptitle("Analytical solution")
    if save_png:
        plt.gcf().savefig(Path(directory, "analytical.png"))

    h_diff, axs1d = Dmp.plotStatic(
        dmp.tau,
        ts,
        xs_ana - xs_ana_cpp,
        xds_ana - xds_ana_cpp,
        forcing_terms=forcing_terms_ana - forcing_terms_ana_cpp,
        fa_outputs=fa_outputs_ana - fa_outputs_ana_cpp,
        plot_tau=False,
    )
    plt.setp(h_diff, linestyle="-", linewidth=1, color=(0.8, 0.2, 0.2))
    plt.gcf().suptitle("Analytical solution (diff)")
    if save_png:
        plt.gcf().savefig(Path(directory, "analytical_diff.png"))

    h_pyt, axs2 = Dmp.plotStatic(dmp.tau, ts, xs_step, xds_step)
    h_cpp, _ = Dmp.plotStatic(dmp.tau, ts, xs_step_cpp, xds_step_cpp, axs=axs2)
    plt.setp(h_pyt, linestyle="-", linewidth=4, color=(0.8, 0.8, 0.8))
    plt.setp(h_cpp, linestyle="--", linewidth=2, color=(0.2, 0.2, 0.8))
    plt.gcf().suptitle("Numerical integration")
    if save_png:
        plt.gcf().savefig(Path(directory, "numerical.png"))

    h_diff, axs2d = Dmp.plotStatic(
        dmp.tau, ts, xs_step - xs_step_cpp, xds_step - xds_step_cpp, plot_tau=False
    )
    plt.setp(h_diff, linestyle="-", linewidth=1, color=(0.8, 0.2, 0.2))
    plt.gcf().suptitle("Numerical integration (diff)")
    if save_png:
        plt.gcf().savefig(Path(directory, "numerical_diff.png"))

    plt.show()
