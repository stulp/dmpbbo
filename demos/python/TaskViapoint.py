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
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
""" Module for the example task TaskViapoint. """

import matplotlib.pyplot as plt
import numpy as np

from dmpbbo.bbo_for_dmps.Task import Task


class TaskViapoint(Task):
    """ Task in which a trajectory has to pass through a viapoint."""

    def __init__(self, viapoint, **kwargs):
        self.viapoint = viapoint
        self.viapoint_time = kwargs.get("viapoint_time", None)
        self.viapoint_radius = kwargs.get("viapoint_radius", 0.0)
        self.goal = kwargs.get("goal", np.zeros(viapoint.shape))
        self.goal_time = kwargs.get("goal_time", None)
        self.viapoint_weight = kwargs.get("viapoint_weight", 1.0)
        self.acceleration_weight = kwargs.get("acceleration_weight", 0.0001)
        self.goal_weight = kwargs.get("goal_weight", 0.0)

        if self.goal is not None:
            if self.goal.shape != self.viapoint.shape:
                raise ValueError("goal and viapoint must have the same shape")

    def get_cost_labels(self):
        """Labels for the different cost components.

        CostFunction.evaluate() may return an array of costs. The first one cost[0] is
        always the sum of the other ones, i.e. costs[0] = sum(costs[1:]). This function
        optionally returns labels for the individual cost components.
        """
        return ["viapoint", "acceleration", "goal"]

    def evaluate_rollout(self, cost_vars, sample):
        """The cost function which defines the task.

        @param cost_vars: All the variables relevant to computing the cost. These are determined by
            TaskSolver.perform_rollout(). For further information see the tutorial on "bbo_for_dmp".
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

        dist_to_viapoint = 0.0
        if self.viapoint_weight > 0.0:

            if self.viapoint_time is None:
                # Don't compute the distance at some time, but rather get the
                # minimum distance

                # Compute all distances along trajectory
                viapoint_repeat = np.repeat(np.atleast_2d(self.viapoint), n_time_steps, axis=0)
                dists = np.linalg.norm(ys - viapoint_repeat, axis=1)

                # Get minimum distance
                dist_to_viapoint = dists.min()  # noqa

            else:
                # Get integer time step at t=viapoint_time
                viapoint_time_step = np.argmax(ts >= self.viapoint_time)
                if viapoint_time_step == 0:
                    print("WARNING: viapoint_time_step=0, maybe viapoint_time is too large?")
                # Compute distance at that time step
                y_via = cost_vars[viapoint_time_step, 1 : 1 + n_dims]
                dist_to_viapoint = np.linalg.norm(y_via - self.viapoint)

            if self.viapoint_radius > 0.0:
                # The viapoint_radius defines a radius within which the cost is
                # always 0
                dist_to_viapoint -= self.viapoint_radius
                if dist_to_viapoint < 0.0:
                    dist_to_viapoint = 0.0

        sum_ydd = 0.0
        if self.acceleration_weight > 0.0:
            sum_ydd = np.sum(np.square(ydds))

        delay_cost_mean = 0.0
        if self.goal_weight > 0.0 and self.goal is not None:
            after_goal_indices = ts >= self.goal_time
            ys_after_goal = ys[after_goal_indices, :]
            n_time_steps = ys_after_goal.shape[0]
            goal_repeat = np.repeat(np.atleast_2d(self.goal), n_time_steps, axis=0)
            delay_cost_mean = np.mean(np.linalg.norm(ys_after_goal - goal_repeat, axis=1))

        costs = np.zeros(1 + 3)
        costs[1] = self.viapoint_weight * dist_to_viapoint
        costs[2] = self.acceleration_weight * sum_ydd / n_time_steps
        costs[3] = self.goal_weight * delay_cost_mean
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
        if n_dims == 1:
            line_handles = ax.plot(t, y, linewidth=0.5)
            ax.plot(t[0], y[0], "bo", label="start")
            ax.plot(t[-1], y[-1], "go", label="end")
            ax.plot(self.viapoint_time, self.viapoint, "ok", label="viapoint")
            if self.viapoint_radius > 0.0:
                r = self.viapoint_radius
                t = self.viapoint_time
                v = self.viapoint[0]
                ax.plot([t, t], [v + r, v - r], "-k")
                ax.set_xlabel("time (s)")
                ax.set_ylabel("y")

        elif n_dims == 2:
            line_handles = ax.plot(y[:, 0], y[:, 1], linewidth=0.5)
            ax.plot(y[0, 0], y[0, 1], "bo", label="start")
            ax.plot(y[-1, 0], y[-1, 1], "go", label="end")
            ax.plot(self.viapoint[0], self.viapoint[1], "ko", label="viapoint")
            if self.viapoint_radius > 0.0:
                circle = plt.Circle(self.viapoint, self.viapoint_radius, color="k", fill=False)
                ax.add_artist(circle)
            ax.axis("equal")
            ax.set_xlabel("y_1")
            ax.set_ylabel("y_2")
        else:
            line_handles = []

        return line_handles, ax
