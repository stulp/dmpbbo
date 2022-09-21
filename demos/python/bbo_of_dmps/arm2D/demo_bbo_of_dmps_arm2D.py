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


import sys

import numpy as np
from matplotlib import pyplot as plt

from demos.python.bbo_of_dmps.TaskViapoint import TaskViapoint
from demos.python.bbo_of_dmps.arm2D.TaskSolverDmpArm2D import TaskSolverDmpArm2D
from demos.python.bbo_of_dmps.arm2D.TaskViapointArm2D import TaskViapointArm2D
from dmpbbo.bbo.DistributionGaussian import DistributionGaussian
from dmpbbo.bbo.updaters import UpdaterCovarAdaptation, UpdaterCovarDecay, UpdaterMean
from dmpbbo.bbo_of_dmps.run_optimization_task import run_optimization_task
from dmpbbo.bbo_of_dmps.TaskSolverDmp import TaskSolverDmp
from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN


def main(directory=None):
    """ Run one demo for bbo_of_dmps.

    @param directory: Directory to save results to
    @param n_dims: Number of dimensions of the task (i.e. the viapoint)
    """

    n_dofs = 7 # Number of joints
    n_dims = 2 # End-effector space dimensionality (must be 2)

    # Prepare a minjerk trajectory in joint angle space
    duration = 0.8
    angles_init = np.full(n_dofs, 0.0)
    angles_goal = np.full(n_dofs, np.pi/n_dofs)
    angles_goal[0] *= 0.5
    ts = np.linspace(0, duration, 51)
    angles_min_jerk = Trajectory.from_min_jerk(ts, angles_init, angles_goal)
    link_lengths = np.full(n_dofs, 1.0 / n_dofs)

    # Train the DMP with the minjerk trajectory
    intersection_height = 0.9
    n_basis = 5
    function_apps = [FunctionApproximatorRBFN(n_basis, intersection_height) for _ in range(n_dofs)]
    dmp = Dmp.from_traj(angles_min_jerk, function_apps)
    dmp.set_selected_param_names("weights")

    # Make task solver, based on a Dmp
    dt = 0.01
    integrate_dmp_beyond_tau_factor = 1.5
    task_solver = TaskSolverDmpArm2D(dmp, 0.01, integrate_dmp_beyond_tau_factor)


    # Make the task
    viapoint = np.full(n_dims, 0.5)

    task = TaskViapointArm2D(
        n_dofs,
        viapoint,
        plot_arm=True,
#        viapoint_time=viapoint_time,
#        viapoint_radius=0.1,
#        goal=y_attr,
#        goal_time=1.1 * tau,
        viapoint_weight=1.0,
        acceleration_weight=0.0001,
#        goal_weight=0.0,
    )


    n_search = dmp.get_param_vector_size()

    mean_init = dmp.get_param_vector()
    #mean_init = np.full(n_search, 0.0)
    covar_init = 100.0 * np.eye(n_search)
    distribution = DistributionGaussian(mean_init, covar_init)

    covar_update = "decay"
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
    n_updates = 30

    session = run_optimization_task(
        task, task_solver, distribution, updater, n_updates, n_samples_per_update, directory
    )
    fig = session.plot()
    fig.canvas.set_window_title(f"Optimization with covar_update={covar_update}")

    plt.show()


if __name__ == "__main__":
    main()
