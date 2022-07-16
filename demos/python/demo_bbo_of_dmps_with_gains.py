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
"""Script for the optimization of DMP trajectories and gain schedules demo."""
import copy
import sys

import numpy as np
from matplotlib import pyplot as plt

from dmpbbo.bbo.DistributionGaussian import DistributionGaussian
from dmpbbo.bbo.updaters import UpdaterCovarAdaptation, UpdaterCovarDecay, UpdaterMean
from dmpbbo.bbo_of_dmps.run_optimization_task import run_optimization_task
from dmpbbo.bbo_of_dmps.Task import Task
from dmpbbo.bbo_of_dmps.TaskSolver import TaskSolver
from dmpbbo.bbo_of_dmps.TaskSolverDmp import TaskSolverDmp
from dmpbbo.dmps.DmpWithSchedules import DmpWithSchedules
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN


class TaskViapointPerturbed(Task):
    """ Task in which a 2D trajectory has to pass through a viapoint. It is perturbed by a force
    field. """

    def __init__(self, viapoint, viapoint_time, min_gain, max_gain, **kwargs):
        self.viapoint = viapoint
        self.viapoint_time = viapoint_time
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.viapoint_weight = kwargs.get("viapoint_weight", 1.0)
        self.acceleration_weight = kwargs.get("acceleration_weight", 0.0001)
        self.gain_weight = kwargs.get("gain_weight", 0.0)

    def get_cost_labels(self):
        """Labels for the different cost components.

        CostFunction.evaluate() may return an array of costs. The first one cost[0] is
        always the sum of the other ones, i.e. costs[0] = sum(costs[1:]). This function
        optionally returns labels for the individual cost components.
        """
        return ["viapoint", "acceleration", "gains"]

    def evaluate_rollout(self, cost_vars, sample):
        """The cost function which defines the task.

        @param cost_vars: All the variables relevant to computing the cost. These are determined by
            TaskSolver.perform_rollout(). For further information see the tutorial on "bbo_of_dmps".
        @param sample: The sample from which the rollout was generated. Passing this to the cost
            function is useful when performing regularization on the sample.
        @return: costs The scalar cost components for the sample. The first item costs[0] should
            contain the total cost.
        """

        n_dims = self.viapoint.shape[0]
        n_time_steps = cost_vars.shape[0]

        ts = cost_vars[:, 0]
        ys = cost_vars[:, 1 : 1 + n_dims]
        ydds = cost_vars[:, 1 + n_dims * 2 : 1 + n_dims * 3]
        gains = cost_vars[:, 1 + n_dims * 3 : 1 + n_dims * 4]

        # Get integer time step at t=viapoint_time
        viapoint_time_step = np.argmax(ts >= self.viapoint_time)
        # Compute distance at that time step

        y_via = ys[viapoint_time_step, :]
        dist_to_viapoint = np.linalg.norm(y_via - self.viapoint)

        costs = np.zeros(1 + 3)
        costs[1] = self.viapoint_weight * dist_to_viapoint
        costs[2] = self.acceleration_weight * np.sum(np.abs(ydds)) / n_time_steps
        costs[3] = self.gain_weight * (
            (np.sum(gains) - self.min_gain) / (n_time_steps * self.max_gain)
        )
        costs[0] = np.sum(costs[1:])
        return costs

    def plot_rollout(self, cost_vars, ax=None):
        """ Plot a rollout (the cost-relevant variables).

        @param cost_vars: Rollout to plot
        @param ax: Axis to plot on (default: None, then a new axis a created)
        @return: line handles and axis
        """

        if not ax:
            ax = plt.axes()

        n_dims = self.viapoint.shape[0]
        t = cost_vars[:, 0]
        y = cost_vars[:, 1 : n_dims + 1]
        gains = cost_vars[:, 1 + n_dims * 3 : 1 + n_dims * 4]
        scaling = 0.5
        line_handles = ax.plot(t, y, linewidth=0.5)
        ax.plot(t, y + scaling * gains, linewidth=0.2)
        ax.plot(t, y - scaling * gains, linewidth=0.2)
        # ax.plot(t[0], y[0], "bo", label="start")
        # ax.plot(t[-1], y[-1], "go", label="end")
        ax.plot(self.viapoint_time, self.viapoint, "ok", label="viapoint")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("y")

        return line_handles, ax


class TaskSolverDmpWithGains(TaskSolver):
    """ TaskSolver that integrates a DMP.

    """

    def __init__(self, dmp_sched, dt, integrate_dmp_beyond_tau_factor):
        self._dmp_sched = copy.deepcopy(dmp_sched)
        self._integrate_time = dmp_sched.tau * integrate_dmp_beyond_tau_factor
        self._n_time_steps = int(np.floor(self._integrate_time / dt)) + 1

    def perform_rollout_dmp_sched(self, dmp_sched):
        """ Perform one rollout for a DMP.

        @param dmp: The DMP to integrate.
        @return: The trajectory generated by the DMP as a matrix.
        """
        ts = np.linspace(0.0, self._integrate_time, self._n_time_steps)
        xs, xds, schedules, forcing_terms, fa_outputs = dmp_sched.analytical_solution_sched(ts)
        traj = dmp_sched.states_as_trajectory(ts, xs, xds)
        traj.misc = schedules
        cost_vars = traj.as_matrix()
        return cost_vars

    def perform_rollout(self, sample, **kwargs):
        """ Perform rollouts, that is, given a set of samples, determine all the variables that
        are relevant to evaluating the cost function.

        @param sample: The sample to perform the rollout for
        @return: The variables relevant to computing the cost.
        """
        self._dmp_sched.set_param_vector(sample)
        return self.perform_rollout_dmp_sched(self._dmp_sched)


def main():
    """ Main function of the script. """
    directory = sys.argv[1] if len(sys.argv) > 1 else None

    # Some DMP parameters
    n_time_steps = 51
    tau = 1.0
    y_init = np.array([0.0])
    y_attr = np.array([1.0])
    n_dims = len(y_init)

    ts = np.linspace(0, tau, n_time_steps)
    traj = Trajectory.from_min_jerk(ts, y_init, y_attr)
    schedule = numpy.full((n_time_steps, n_dims), 0.1)
    traj.misc = schedule

    function_apps = [FunctionApproximatorRBFN(8, 0.95) for _ in range(n_dims)]
    function_apps_schedules = [FunctionApproximatorRBFN(5, 0.95) for _ in range(n_dims)]
    dmp = DmpWithSchedules.from_traj_sched(traj, function_apps, function_apps_schedules)

    # xs, xds, sched, _, _ = dmp.analytical_solution_sched(ts)
    # dmp.plot_sched(ts, xs, xds, sched)
    # plt.show()

    dmp.set_selected_param_names(["weights"])
    n_search_traj = dmp.get_param_vector_size()
    dmp.set_selected_param_names(["sched_weights"])
    n_search_gains = dmp.get_param_vector_size()
    dmp.set_selected_param_names(["weights", "sched_weights"])

    # Make the task
    task = TaskViapointPerturbed(
        numpy.full((1, n_dims), 0.6),
        0.5,
        1.0,
        10.0,
        viapoint_weight=1.0,
        acceleration_weight=0.1,
        gain_weight=0.1,
    )

    # Make task solver, based on a Dmp
    dt = 0.01
    integrate_dmp_beyond_tau_factor = 1.5
    task_solver = TaskSolverDmpWithGains(dmp, dt, integrate_dmp_beyond_tau_factor)

    mean_init = dmp.get_param_vector()
    sigma_traj = 10
    sigma_gains = 1
    sigmas = np.concatenate(
        (np.full(n_search_traj, sigma_traj), np.full(n_search_gains, sigma_gains))
    )
    covar_init = np.diag(np.square(sigmas))
    distribution = DistributionGaussian(mean_init, covar_init)
    updater = UpdaterCovarDecay(eliteness=10, weighting_method="PI-BB", covar_decay_factor=0.85)

    n_samples_per_update = 10
    n_updates = 10

    session = run_optimization_task(
        task, task_solver, distribution, updater, n_updates, n_samples_per_update, directory
    )
    fig = session.plot()
    plt.show()


if __name__ == "__main__":
    main()
