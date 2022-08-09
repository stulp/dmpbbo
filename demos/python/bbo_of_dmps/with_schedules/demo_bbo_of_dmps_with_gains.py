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
import random
import sys

import numpy as np
from matplotlib import pyplot as plt

from demos.python.bbo_of_dmps.with_schedules import force_field_simulator
from dmpbbo.bbo.DistributionGaussian import DistributionGaussian
from dmpbbo.bbo.updaters import UpdaterCovarDecay
from dmpbbo.bbo_of_dmps.run_optimization_task import run_optimization_task
from dmpbbo.bbo_of_dmps.Task import Task
from dmpbbo.bbo_of_dmps.TaskSolver import TaskSolver
from dmpbbo.dmps.DmpWithSchedules import DmpWithSchedules
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN


class TaskViapointWithGains(Task):
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

        # cost_vars:
        #   0  1      2       3        4      5      6       7
        #   t, y_cur, yd_cur, ydd_cur, gains, y_des, yd_des, ydd_des
        ts = cost_vars[:, 0]
        ys_cur = cost_vars[:, 1 : 1 + n_dims]
        ydds_cur = cost_vars[:, 1 + n_dims * 2 : 1 + n_dims * 3]
        gains = cost_vars[:, 1 + n_dims * 3 : 1 + n_dims * 4]

        # Get integer time step at t=viapoint_time
        viapoint_time_step = np.argmax(ts >= self.viapoint_time)
        # Compute distance at that time step
        y_via = ys_cur[viapoint_time_step, :]
        dist_to_viapoint = np.linalg.norm(y_via - self.viapoint)

        costs = np.zeros(1 + 3)
        costs[1] = self.viapoint_weight * dist_to_viapoint
        costs[2] = self.acceleration_weight * np.sum(np.abs(ydds_cur)) / n_time_steps
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

        # cost_vars:
        #   0  1      2       3        4      5      6       7
        #   t, y_cur, yd_cur, ydd_cur, gains, y_des, yd_des, ydd_des
        n_dims = self.viapoint.shape[0]
        t = cost_vars[:, 0]
        y_cur = cost_vars[:, 1 : n_dims + 1]
        gains = cost_vars[:, 1 + n_dims * 3 : 1 + n_dims * 4]
        y_des = cost_vars[:, 1 + n_dims * 4 : 1 + n_dims * 5]
        scaling = 1.0
        line_handles = []
        lh1 = ax.plot(t, y_des, "--", linewidth=0.4)
        line_handles.extend(lh1)
        # lh2 = ax.plot(t, y_cur, linewidth=0.5)
        # line_handles.extend(lh2)
        ax.set_ylim([-1.5, 1.5])
        # ax.plot(t, y_des + scaling * gains, linewidth=0.2)
        # ax.plot(t, y_des - scaling * gains, linewidth=0.2)
        ax2 = ax.twinx()
        lh3 = ax2.plot(t, scaling * gains, linewidth=1)
        ax2.set_ylim([0, 1000])
        line_handles.extend(lh3)

        # ax.plot(t[0], y[0], "bo", label="start")
        # ax.plot(t[-1], y[-1], "go", label="end")
        ax.plot(self.viapoint_time, self.viapoint, "ok", label="viapoint")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("y")

        return line_handles, ax


class TaskSolverDmpWithGainsAndForceField(TaskSolver):
    """ TaskSolver that integrates a DMP and applies a force field

    """

    def __init__(self, dmp_sched, dt, integrate_dmp_beyond_tau_factor, stochastic_field):
        self._dmp_sched = copy.deepcopy(dmp_sched)
        self._integrate_time = dmp_sched.tau * integrate_dmp_beyond_tau_factor
        self._n_time_steps = int(np.floor(self._integrate_time / dt)) + 1
        self.stochastic_field = stochastic_field

    def perform_rollout_dmp_sched(self, dmp_sched):
        """ Perform one rollout for a DMP.

        @param dmp_sched: The DMP to integrate.
        @return: The trajectory generated by the DMP as a matrix.
        """

        field_strength = -20.0
        if self.stochastic_field:
            field_strength = random.uniform(-20.0, 20.0)

        field_max_time = 0.5 * dmp_sched.tau
        r = force_field_simulator.perform_rollout(
            dmp_sched, self._integrate_time, self._n_time_steps, field_strength, field_max_time
        )

        cost_vars = np.column_stack(
            (
                r["ts"],
                r["ys_cur"],
                r["yds_cur"],
                r["ydds_cur"],
                r["schedules"],
                r["ys_des"],
                r["yds_des"],
                r["ydds_des"],
            )
        )

        return cost_vars
        # return r

    def perform_rollout(self, sample, **kwargs):
        """ Perform rollouts, that is, given a set of samples, determine all the variables that
        are relevant to evaluating the cost function.

        @param sample: The sample to perform the rollout for
        @return: The variables relevant to computing the cost.
        """
        self._dmp_sched.set_param_vector(sample)
        return self.perform_rollout_dmp_sched(self._dmp_sched)


def run_optimization(stochastic_field, directory=None):

    # Main parameter of the experiment
    gain_min = 10.0
    gain_max = 1000.0
    gain_initial = 10.0

    # Some DMP parameters
    n_time_steps = 51
    tau = 1.0
    y_init = np.array([0.0])
    y_attr = np.array([1.0])
    n_dims = len(y_init)
    ts = np.linspace(0, tau, n_time_steps)

    # Train the DMP from a min-jerk trajectory, and constant gains
    traj = Trajectory.from_min_jerk(ts, y_init, y_attr)
    schedule = np.full((n_time_steps, n_dims), gain_initial)
    traj.misc = schedule
    function_apps = [FunctionApproximatorRBFN(7, 0.95) for _ in range(n_dims)]
    function_apps_schedules = [FunctionApproximatorRBFN(5, 0.9) for _ in range(n_dims)]
    dmp = DmpWithSchedules.from_traj_sched(
        traj, function_apps, function_apps_schedules, min_schedules=gain_min, max_schedules=gain_max
    )

    # xs, xds, sched, _, _ = dmp.analytical_solution_sched(ts)
    # dmp.plot_sched(ts, xs, xds, sched)
    # plt.show()

    # Determine the size of the search space for DMP weights and gain schedules
    dmp.set_selected_param_names(["weights"])
    n_search_traj = dmp.get_param_vector_size()
    dmp.set_selected_param_names(["sched_weights"])
    n_search_gains = dmp.get_param_vector_size()
    # We know the search spaces now: optimize both.
    dmp.set_selected_param_names(["weights", "sched_weights"])

    # Make the task
    viapoint = np.full((1, n_dims), 0.5)
    viapoint_time = 0.5
    task = TaskViapointWithGains(
        viapoint,
        viapoint_time,
        gain_min,
        gain_max,
        viapoint_weight=3.0,
        acceleration_weight=0.0,
        gain_weight=1.0,
    )

    # Make task solver, based on a Dmp
    dt = 0.05
    integrate_dmp_beyond_tau_factor = 1.2
    task_solver = TaskSolverDmpWithGainsAndForceField(
        dmp, dt, integrate_dmp_beyond_tau_factor, stochastic_field
    )

    # Determine the initial covariance matrix, and its updater
    mean_init = dmp.get_param_vector()
    sigma_traj = 10.0
    sigma_gains = 50.0
    sigmas = np.concatenate(
        (np.full(n_search_traj, sigma_traj), np.full(n_search_gains, sigma_gains))
    )
    covar_init = np.diag(np.square(sigmas))
    distribution = DistributionGaussian(mean_init, covar_init)
    updater = UpdaterCovarDecay(eliteness=10, weighting_method="PI-BB", covar_decay_factor=0.98)

    n_samples_per_update = 10
    n_updates = 50
    session = run_optimization_task(
        task, task_solver, distribution, updater, n_updates, n_samples_per_update, directory
    )

    ax = plt.figure(figsize=(5, 5)).add_subplot(1, 1, 1)
    for i_update in range(n_updates):
        handle, _ = session.plot_rollouts_update(i_update, ax=ax)
        session._set_style(handle, i_update, n_updates)

    window_label = "stochastic" if stochastic_field else "constant"
    plt.gcf().canvas.set_window_title(window_label + " force field")
    return session


def main():
    """ Main function of the script. """
    directory = sys.argv[1] if len(sys.argv) > 1 else None
    for stochastic_field in [False, True]:
        session = run_optimization(stochastic_field, directory)
        fig = plt.figure(figsize=(20, 5))
        session.plot(fig)

    plt.show()


if __name__ == "__main__":
    main()
