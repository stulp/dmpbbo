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
""" Module for the example task TaskViapoint. """

import matplotlib.pyplot as plt
import numpy as np

from demos.python.bbo_of_dmps.TaskViapoint import TaskViapoint
from dmpbbo.bbo_of_dmps.Task import Task
from dmpbbo.dmps.Trajectory import Trajectory


class TaskViapointArm2D(TaskViapoint):
    """ Task in which a trajectory has to pass through a viapoint."""

    def __init__(self, n_dofs, viapoint, **kwargs):
        super().__init__(viapoint, **kwargs)
        self.n_dofs = n_dofs
        self.plot_arm = kwargs.get("plot_arm", True)

    def evaluate_rollout(self, cost_vars, sample):
        """The cost function which defines the task.

        @param cost_vars: All the variables relevant to computing the cost. These are determined by
            TaskSolver.perform_rollout(). For further information see the tutorial on "bbo_of_dmps".
        @param sample: The sample from which the rollout was generated. Passing this to the cost
            function is useful when performing regularization on the sample.
        @return: costs The scalar cost components for the sample. The first item costs[0] should
            contain the total cost.
        """
        n_cost_vars = 1 + self.n_dofs*3 + (self.n_dofs+1)*2
        assert(n_cost_vars == cost_vars.shape[1])

        ts = cost_vars[:, 0]
        n_link_pos = (self.n_dofs + 1) * 2
        link_positions = cost_vars[:, -n_link_pos:]
        n_dims = 2 # Per definition, see the name of the class
        endeff_positions = cost_vars[:, -n_dims:]

        offset = 1 + self.n_dofs*2
        joint_accelerations = cost_vars[:, offset:offset+self.n_dofs]

        return super().evaluate_rollout_local(ts, endeff_positions, joint_accelerations, sample)


    def plot_rollout(self, cost_vars, ax=None):
        """ Plot a rollout (the cost-relevant variables).

        @param cost_vars: Rollout to plot
        @param ax: Axis to plot on (default: None, then a new axis a created)
        @return: line handles and axis
        """
        # 1 => time, n_dofs*3 => y/yd/ydd for each joint, (n_dofs+1)*2 => 2D pos of each link
        n_cost_vars = 1 + self.n_dofs*3 + (self.n_dofs+1)*2
        assert(n_cost_vars == cost_vars.shape[1])
        ts = cost_vars[:, 0]
        n_link_pos = (self.n_dofs + 1) * 2
        link_positions = cost_vars[:, -n_link_pos:]
        n_plots = 20
        line_handles, ax = TaskViapointArm2D.plot_link_positions(link_positions, n_plots, ax,
                                                 plot_arm=self.plot_arm)

        n_dims = 2
        end_eff_positions = link_positions[:,-n_dims:]
        _, y_waypoint = super().get_waypoint(ts, end_eff_positions)
        color = line_handles[0].get_color()

        ax.plot(self.viapoint[0], self.viapoint[1], "ko", label="viapoint")
        ax.plot([y_waypoint[0], self.viapoint[0]], [y_waypoint[1], self.viapoint[1]], "ko-",
                label="waypoint", markerfacecolor='none')
        return line_handles, ax

    @staticmethod
    def plot_link_positions(links_pos, n_plots=10, ax=None, **kwargs):
        if not ax:
            ax = plt.axes()
        plot_arm = kwargs.get("plot_arm", True)
        links_x = links_pos[:, 0::2]
        links_y = links_pos[:, 1::2]
        line_handles = ax.plot(links_x[:, -1], links_y[:, -1], 'r-')
        if plot_arm:
            n_time_steps = links_pos.shape[0]
            n_plots = np.min([n_plots, n_time_steps])
            for tt in np.linspace(0, n_time_steps - 1, n_plots):
                h = ax.plot(links_x[int(tt), :], links_y[int(tt), :], '-', color='#cccccc')
                line_handles.extend(h)
        ax.axis('equal')
        return line_handles, ax
