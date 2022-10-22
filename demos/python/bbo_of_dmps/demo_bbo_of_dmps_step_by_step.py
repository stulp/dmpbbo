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
"""Script for bbo_of_dmps demo (with single updates saved to file)."""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from dmpbbo.bbo.DistributionGaussian import DistributionGaussian
from dmpbbo.bbo.updaters import UpdaterCovarDecay
from dmpbbo.bbo_of_dmps.step_by_step_optimization import prepare_optimization, update_step
from dmpbbo.bbo_of_dmps.TaskSolverDmp import TaskSolverDmp
from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dmps.Dmp import Trajectory
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
    plot_dmp = False
    if plot_dmp:
        dt = 0.005
        integration_time =  3*dmp.tau
        ts = np.linspace(0.0, integration_time, int(integration_time / dt) + 1)
        dmp.plot(ts)  # Check if all is OK with the dynamical systems
        dmp.plot_comparison(traj)  # Check if fitting is good.

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
    dt = 0.005
    integrate_dmp_beyond_tau_factor = 1.5
    task_solver = TaskSolverDmp(dmp, dt, integrate_dmp_beyond_tau_factor)

    dmp.set_selected_param_names("weights")
    # dmp.set_selected_param_names(['goal','weights'])
    mean_init = dmp.get_param_vector()
    covar_init = 100.0 * np.eye(mean_init.size)
    distribution = DistributionGaussian(mean_init, covar_init)

    updater = UpdaterCovarDecay(eliteness=10, weighting_method="PI-BB", covar_decay_factor=0.9)

    n_samples_per_update = 10
    n_updates = 20

    session = prepare_optimization(
        directory, task, task_solver, distribution, n_samples_per_update, updater, dmp
    )

    for i_update in range(n_updates):
        dmp_eval = session.ask("dmp", i_update, "eval")
        cost_vars_eval = task_solver.perform_rollout_dmp(dmp_eval)
        session.tell(cost_vars_eval, "cost_vars", i_update, "eval")

        for i_sample in range(n_samples_per_update):
            dmp_sample = session.ask("dmp", i_update, i_sample)
            cost_vars = task_solver.perform_rollout_dmp(dmp_sample)
            session.tell(cost_vars, "cost_vars", i_update, i_sample)

        update_step(session, i_update)

    return session.plot()


def main():
    """ Main function of the script. """
    default_dir = Path(tempfile.gettempdir(), "dmpbbo", "demo_bbo_of_dmps_step_by_step")

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
        for n_dims in [2]:
            tau = 0.5
            dt = 0.005
            ts = np.linspace(0.0, tau, int(tau/dt)+1)
            y_init = np.linspace(1.8, 2.0, n_dims)
            y_attr = np.linspace(4.0, 3.0, n_dims)
            traj = Trajectory.from_min_jerk(ts, y_init, y_attr)
            run_demo(args.dir, traj)

    plt.show()


if __name__ == "__main__":
    main()
