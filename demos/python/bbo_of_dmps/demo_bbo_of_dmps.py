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
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
"""Script for bbo_of_dmps demo."""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from dmpbbo.bbo.DistributionGaussian import DistributionGaussian
from dmpbbo.bbo.updaters import UpdaterCovarAdaptation, UpdaterCovarDecay, UpdaterMean
from dmpbbo.bbo_of_dmps.run_optimization_task import run_optimization_task
from dmpbbo.bbo_of_dmps.TaskSolverDmp import TaskSolverDmp
from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN
from TaskViapoint import TaskViapoint


def run_demo(directory, traj):
    """ Run one demo for bbo_of_dmps (with single updates)

    @param directory: Directory to save results to
    @param traj: Initial trajectory
    """

    n_dims = traj.dim
    n_basis = 10
    h = 0.9  # intersection_height
    function_apps = [FunctionApproximatorRBFN(n_basis, h) for _ in range(n_dims)]

    dmp = Dmp.from_traj(traj, function_apps)
    dmp.set_selected_param_names("weights")
    # dmp.set_selected_param_names(['goal','weights'])

    # Make the task
    viapoint = 3 * np.ones(n_dims)
    viapoint_time = (
        0.3 if n_dims == 1 else None
    )  # None means: Do not pass through viapoint at a specific time,
    # but rather pass through it at any time.

    task = TaskViapoint(
        viapoint,
        viapoint_time=viapoint_time,
        viapoint_radius=0.1,
        goal=dmp.y_attr,
        goal_time=1.1 * dmp.tau,
        viapoint_weight=1.0,
        acceleration_weight=0.00005,
        goal_weight=0.0,
    )

    # Make task solver, based on a Dmp
    dt = 0.01
    integrate_dmp_beyond_tau_factor = 1.5
    task_solver = TaskSolverDmp(dmp, dt, integrate_dmp_beyond_tau_factor)

    n_search = dmp.get_param_vector_size()

    mean_init = np.full(n_search, 0.0)
    covar_init = 1000.0 * np.eye(n_search)
    distribution = DistributionGaussian(mean_init, covar_init)

    covar_update = "cma"
    if covar_update == "none":
        updater = UpdaterMean(eliteness=10, weighting_method="PI-BB")
    elif covar_update == "decay":
        updater = UpdaterCovarDecay(eliteness=10, weighting_method="PI-BB", covar_decay_factor=0.9)
    else:
        updater = UpdaterCovarAdaptation(
            eliteness=10,
            weighting_method="PI-BB",
            max_level=None,
            min_level=1.0,
            diag_only=False,
            learning_rate=0.5,
        )

    n_samples_per_update = 10
    n_updates = 40

    session = run_optimization_task(
        task, task_solver, distribution, updater, n_updates, n_samples_per_update, directory
    )
    fig = session.plot()
    fig.canvas.set_window_title(f"Optimization with covar_update={covar_update}")


def main():
    """ Main function of the script. """
    default_dir = Path(tempfile.gettempdir(), "dmpbbo", "demo_bbo_of_dmps")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="dir to write results to", default=default_dir)
    parser.add_argument("--traj", type=str, help="optional init trajectory file")
    args = parser.parse_args()

    if args.traj:
        traj = Trajectory.loadtxt(args.traj)
        # traj.plot # Check if trajectory loaded correctly.
        run_demo(args.dir, traj)

    else:
        # Some DMP parameters
        for n_dims in [1, 2]:
            tau = 0.5
            dt = 0.005
            ts = np.linspace(0.0, tau, int(tau / dt) + 1)
            y_init = np.linspace(1.8, 2.0, n_dims)
            y_attr = np.linspace(4.0, 3.0, n_dims)
            traj = Trajectory.from_min_jerk(ts, y_init, y_attr)
            run_demo(args.dir, traj)

    plt.show()


if __name__ == "__main__":
    main()
