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

from dmpbbo.bbo_of_dmps.Task import Task


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
        self.regularization_weight = kwargs.get("regularization_weight", 0.0)
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
        return ["viapoint", "acceleration", "goal", "regularization"]

    def get_waypoint(self, ts, ys):
        if self.viapoint_time is None:
            # Don't compute the distance at some time, but rather get the
            # minimum distance

            # Compute all distances along trajectory
            n_time_steps = len(ts)
            viapoint_repeat = np.repeat(np.atleast_2d(self.viapoint), n_time_steps, axis=0)
            dists = np.linalg.norm(ys - viapoint_repeat, axis=1)

            # Get the index of the point with the smallest distance
            i_waypoint = dists.argmin()  # noqa
            # This is the waypoint
            y_waypoint = ys[i_waypoint]
            t_waypoint = ts[i_waypoint]
            return t_waypoint, y_waypoint

        else:
            # Get integer time step at t=viapoint_time
            viapoint_time_step = np.argmax(ts >= self.viapoint_time)
            if viapoint_time_step == 0:
                print("WARNING: viapoint_time_step=0, maybe viapoint_time is too large?")
            # Compute distance at that time step
            y_waypoint = ys[viapoint_time_step, :]
            return self.viapoint_time, y_waypoint

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
        ts = cost_vars[:, 0]
        ys = cost_vars[:, 1 : 1 + n_dims]
        ydds = cost_vars[:, 1 + n_dims * 2 : 1 + n_dims * 3]
        return self.evaluate_rollout_local(ts, ys, ydds, sample)


    def evaluate_rollout_local(self, ts, ys, ydds, sample):

        n_dims = self.viapoint.shape[0]
        n_time_steps = ts.shape[0]

        dist_to_viapoint = 0.0
        if self.viapoint_weight > 0.0:

            _, y_waypoint = self.get_waypoint(ts, ys)
            dist_to_viapoint = np.linalg.norm(y_waypoint - self.viapoint)

            if self.viapoint_radius > 0.0:
                # The viapoint_radius defines a radius within which the cost is
                # always 0
                dist_to_viapoint -= self.viapoint_radius
                if dist_to_viapoint < 0.0:
                    dist_to_viapoint = 0.0

        sum_ydd = 0.0
        if self.acceleration_weight > 0.0:
            sum_ydd = np.sum(np.square(ydds))
            if ydds.ndim > 1:
                # Divide by number of joints/dimensions to make invariant to dimensionality.
                sum_ydd /= ydds.shape[1]

        l2_norm = 0.0
        if self.regularization_weight > 0.0:
            l2_norm = np.sqrt(np.sum(np.square(sample)))

        delay_cost_mean = 0.0
        if self.goal_weight > 0.0 and self.goal is not None:
            after_goal_indices = ts >= self.goal_time
            ys_after_goal = ys[after_goal_indices, :]
            n_time_steps = ys_after_goal.shape[0]
            goal_repeat = np.repeat(np.atleast_2d(self.goal), n_time_steps, axis=0)
            delay_cost_mean = np.mean(np.linalg.norm(ys_after_goal - goal_repeat, axis=1))


        costs = np.zeros(1 + 4)
        costs[1] = self.viapoint_weight * dist_to_viapoint
        costs[2] = self.acceleration_weight * sum_ydd / n_time_steps
        costs[3] = self.goal_weight * delay_cost_mean
        costs[4] = self.regularization_weight * l2_norm
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
        ts = cost_vars[:, 0]
        ys = cost_vars[:, 1 : n_dims + 1]
        if n_dims == 1:
            line_handles = ax.plot(ts, ys, linewidth=0.5)
            ax.plot(ts[0], ys[0], "bo", label="start")
            ax.plot(ts[-1], ys[-1], "go", label="end")
            if self.viapoint_time:
                ax.plot(self.viapoint_time, self.viapoint, "ok", label="viapoint")
            if self.viapoint_radius > 0.0:
                r = self.viapoint_radius
                t = self.viapoint_time
                v = self.viapoint[0]
                ax.plot([t, t], [v + r, v - r], "-k")
            t_waypoint, y_waypoint = self.get_waypoint(ts,ys)
            ax.plot([t_waypoint, t_waypoint], [y_waypoint,  self.viapoint], "ko-",
                    label="waypoint",markerfacecolor='none')
            ax.set_xlabel("time (s)")
            ax.set_ylabel("y")

        elif n_dims == 2:
            line_handles = ax.plot(ys[:, 0], ys[:, 1], linewidth=0.5)
            ax.plot(ys[0, 0], ys[0, 1], "bo", label="start")
            ax.plot(ys[-1, 0], ys[-1, 1], "go", label="end")
            ax.plot(self.viapoint[0], self.viapoint[1], "ko", label="viapoint")
            _, y_waypoint = self.get_waypoint(ts,ys)
            ax.plot([y_waypoint[0], self.viapoint[0]], [y_waypoint[1], self.viapoint[1]], "ko-",
                    label="waypoint",markerfacecolor='none')

            if self.viapoint_radius > 0.0:
                circle = plt.Circle(self.viapoint, self.viapoint_radius, color="k", fill=False)
                ax.add_artist(circle)
            ax.axis("equal")
            ax.set_xlabel("y_1")
            ax.set_ylabel("y_2")
        else:
            line_handles = []

        return line_handles, ax

def main():
    """ Main function of the script. """
    from dmpbbo.dmps.Trajectory import Trajectory # Only needed when main is called

    duration = 0.8
    viapoint_time = 0.5*duration
    viapoint_time = None # Finds the closest waypoint, rather than at a specific time
    n_dims = 2
    viapoint = np.full(n_dims, 0.5)

    task = TaskViapoint(viapoint, viapoint_time=viapoint_time)

    ts = np.linspace(0, duration, 201)
    y_init = np.full(n_dims, 0.0)
    y_goal_mean = np.full(n_dims, 1.0)
    for _ in range(5):
        y_goal = y_goal_mean + 0.3*np.random.randn(n_dims)
        traj_min_jerk = Trajectory.from_min_jerk(ts, y_init, y_goal)
        cost_vars = traj_min_jerk.as_matrix()

        costs = task.evaluate_rollout(cost_vars, y_goal)
        cost_labels = task.get_cost_labels()

        _, ax = task.plot_rollout(traj_min_jerk.as_matrix())
        if n_dims==1:
            ax.text(duration,y_goal, f'{np.array2string(costs, precision=3)}')
        else:
            ax.text(y_goal[0], y_goal[1], f'{np.array2string(costs, precision=3)}')
    plt.show()


if __name__ == "__main__":
    main()
