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

import numpy as np
from matplotlib import pyplot as plt

import dmpbbo.json_for_cpp as jc
from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN
from execute_binary import execute_binary


def main(show=False, save=False, verbose=False):
    directory = "/tmp/testDmp/"
    os.makedirs(directory, exist_ok=True)

    ################################
    # Read trajectory and train DMP with it.
    trajectory_file = Path("..", "fixtures", "trajectory.txt")
    if verbose:
        print(f"Reading trajectory from: {trajectory_file}\n")
    traj = Trajectory.loadtxt(trajectory_file)
    n_dims = traj.dim

    n_bfs = 10
    function_apps = [FunctionApproximatorRBFN(n_bfs, 0.7) for _ in range(n_dims)]
    dmp = Dmp.from_traj(traj, function_apps, dmp_type="KULVICIUS_2012_JOINING")

    ################
    # Analytical solution to compute difference
    if verbose:
        print("===============\nPython Analytical solution")

    ts = traj.ts
    xs_ana, xds_ana, forcing_terms_ana, fa_outputs_ana = dmp.analytical_solution(ts)

    ################################################
    # Numerically integrate the DMP

    if verbose:
        print("===============\nPython Numerical integration")
    n_time_steps = len(ts)
    dim_x = xs_ana.shape[1]
    xs_step = np.zeros([n_time_steps, dim_x])
    xds_step = np.zeros([n_time_steps, dim_x])

    (x, xd) = dmp.integrate_start()
    xs_step[0, :] = x
    xds_step[0, :] = xd
    for tt in range(1, n_time_steps):
        dt = ts[tt] - ts[tt - 1]
        xs_step[tt, :], xds_step[tt, :] = dmp.integrate_step_runge_kutta(dt, xs_step[tt - 1, :])

    ################################################
    # Call the binary, which does analytical_solution and numerical integration in C++

    # Save the dynamical system to a json file
    jc.savejson(Path(directory, "dmp.json"), dmp)
    jc.savejson_for_cpp(Path(directory, "dmp_for_cpp.json"), dmp)
    np.savetxt(Path(directory, "ts.txt"), ts)

    exec_name = "../../bin/testDmp"
    execute_binary(exec_name, f"{directory} dmp")

    if verbose:
        print("===============\nPython reading output from C++")
    d = directory
    xs_ana_cpp = np.loadtxt(os.path.join(d, "xs_ana.txt"))
    xds_ana_cpp = np.loadtxt(os.path.join(d, "xds_ana.txt"))
    forcing_terms_ana_cpp = np.loadtxt(os.path.join(d, "forcing_terms_ana.txt"))
    fa_outputs_ana_cpp = np.loadtxt(os.path.join(d, "fa_outputs_ana.txt"))
    xs_step_cpp = np.loadtxt(os.path.join(d, "xs_step.txt"))
    xds_step_cpp = np.loadtxt(os.path.join(d, "xds_step.txt"))

    # Plotting

    if verbose:
        print("===============\nPython Plotting")
    h_pyt, axs1 = dmp.plot(
        ts, xs_ana, xds_ana, forcing_terms=forcing_terms_ana, fa_outputs=fa_outputs_ana
    )
    h_cpp, _ = dmp.plot(
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

    if save:
        plt.gcf().savefig(Path(directory, "analytical.png"))

    h_diff, axs1d = dmp.plot(
        ts,
        xs_ana - xs_ana_cpp,
        xds_ana - xds_ana_cpp,
        forcing_terms=forcing_terms_ana - forcing_terms_ana_cpp,
        fa_outputs=fa_outputs_ana - fa_outputs_ana_cpp,
        plot_tau=False,
    )
    plt.setp(h_diff, linestyle="-", linewidth=1, color=(0.8, 0.2, 0.2))
    plt.gcf().suptitle("Analytical solution (diff)")
    if save:
        plt.gcf().savefig(Path(directory, "analytical_diff.png"))

    h_pyt, axs2 = dmp.plot(ts, xs_step, xds_step)
    h_cpp, _ = dmp.plot(ts, xs_step_cpp, xds_step_cpp, axs=axs2)
    plt.setp(h_pyt, linestyle="-", linewidth=4, color=(0.8, 0.8, 0.8))
    plt.setp(h_cpp, linestyle="--", linewidth=2, color=(0.2, 0.2, 0.8))
    plt.gcf().suptitle("Numerical integration")
    if save:
        plt.gcf().savefig(Path(directory, "numerical.png"))

    h_diff, axs2d = dmp.plot(ts, xs_step - xs_step_cpp, xds_step - xds_step_cpp, plot_tau=False)
    plt.setp(h_diff, linestyle="-", linewidth=1, color=(0.8, 0.2, 0.2))
    plt.gcf().suptitle("Numerical integration (diff)")
    if save:
        plt.gcf().savefig(Path(directory, "numerical_diff.png"))

    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="show plots")
    parser.add_argument("--save", action="store_true", help="save plots")
    parser.add_argument("--verbose", action="store_true", help="print output")
    args = parser.parse_args()

    main(args.show, args.save, args.verbose)
