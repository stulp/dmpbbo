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
""" Module for the LearningSessionTask class. """

import inspect
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import dmpbbo.json_for_cpp as jc
from dmpbbo.bbo.LearningSession import LearningSession


class LearningSessionTask(LearningSession):
    """ Database for storing information about learning progress for specific task (Task)
    """

    def __init__(self, n_samples_per_update, directory=None, **kwargs):
        super().__init__(n_samples_per_update, directory, **kwargs)
        # self._task_solver = kwargs.get("task_solver", None)
        task = kwargs.get("task", None)
        if directory and task:
            src = inspect.getsourcelines(task.__class__)
            src = " ".join(src[0])
            src = src.replace("(Task)", "")
            filename = Path(directory, "task.py")
            with open(filename, "w") as f:
                f.write(src)

    def tell(self, obj, name, i_update=None, i_sample=None):
        """ Add an object to the database.

        @param obj:  The object to add
        @param name:  The name of the file
        @param i_update:  The update number
        @param i_sample:  The sample number
        """
        # If it's a Dmp, save it in a C++-readable format also
        if "dmp" in name:
            if self._root_dir is not None:
                basename = self.get_base_name(name, i_update, i_sample)
                abs_basename = Path(self._root_dir, basename)
                jc.savejson(f"{abs_basename}.json", obj)
                jc.savejson_for_cpp(f"{abs_basename}_for_cpp.json", obj)

        filename = super().tell(obj, name, i_update, i_sample)
        return filename

    def add_rollout(self, i_update, i_sample, sample, cost_vars, cost):
        """

        @param i_update: The update number
        @param i_sample:  The sample number
        @param sample: The sample for this rollout
        @param cost_vars: The cost-relevant variables for the rollout.
        @param cost: The cost of the rollout
        """
        self.tell(sample, "sample", i_update, i_sample)
        self.tell(cost_vars, "cost_vars", i_update, i_sample)
        self.tell(cost, "cost", i_update, i_sample)

    def add_eval_task(self, i_update, eval_sample, eval_cost_vars, eval_cost):
        """ Add an evaluation of the task

        @param i_update: The update number for which this is the evaluation
        @param eval_cost_vars: The cost-relevant variables for the evaluation.
        @param eval_sample: The sample for which the evaluation was made.
        @param eval_cost: The cost of the evaluation sample
        """
        super().add_eval(i_update, eval_sample, eval_cost)
        self.tell(eval_cost_vars, "eval_cost_vars", i_update)

    @staticmethod
    def _set_style(handle, i_update, n_updates):
        """ Set the color of an object, according to how far the optimization has proceeded.

        @param handle: Handle to the object
        @param i_update: Which update is the optimization at now?
        @param n_updates: Which is the number of updates the optimization will run?
        """
        if i_update == 0:
            plt.setp(handle, color=[0.8, 0.0, 0.0], linewidth=2)
        else:
            c = 1.0 * i_update / n_updates
            cur_color = [0.3 - 0.3 * c, 0.0 + 1.0 * c, 0.0 - 0.0 * c]
            plt.setp(handle, color=cur_color, linewidth=2)

    def plot_rollouts(self, ax=None, max_n_updates=20):
        """ Plot all rollouts during the learning session

        @param ax:  Axis to plot on (default: None, then a new axis is initialized)
        @param max_n_updates:  Max number of updates (lines) to plot
        @return: the line handles and the axis handle
        """
        if not ax:
            ax = plt.axes()
        all_lines = []
        n_updates = self.get_n_updates()

        if n_updates > max_n_updates:
            updates_list = [int(np.round(i)) for i in np.linspace(0, n_updates, max_n_updates)]
        else:
            updates_list = list(range(n_updates+1))

        for i_update in updates_list:
            lines, _ = self.plot_rollouts_update(
                i_update, ax=ax, plot_eval=True, plot_samples=False
            )
            LearningSessionTask._set_style(lines, i_update, n_updates)
            all_lines.extend(lines)
        return all_lines, ax

    def plot_rollouts_update(self, i_update, **kwargs):
        """ Plot the rollouts for one update

        @param i_update: The update number to plot
        @return: the line handles and the axis handle
        """
        ax = kwargs.get("ax", None) or plt.axes()
        plot_eval = kwargs.get("plot_eval", True)
        plot_samples = kwargs.get("plot_samples", False)

        task = self.ask("task")
        lines_eval = []
        if plot_eval and self.exists("cost_vars", i_update, "eval"):
            cost_vars = self.ask("cost_vars", i_update, "eval")
            if task:
                lines_eval, _ = task.plot_rollout(cost_vars, ax)
            else:
                lines_eval = ax.plot(cost_vars)
                if not isinstance(lines_eval, list):
                    lines_eval = [lines_eval]
            plt.setp(lines_eval, color="#3333ff", linewidth=3)

        if plot_samples:
            n_samples = self.ask("n_samples_per_update")
            for i_sample in range(n_samples):
                cost_vars = self.ask("cost_vars", i_update, i_sample)
                if task:
                    lines, _ = task.plot_rollout(cost_vars, ax)
                else:
                    lines = ax.plot(cost_vars)
                plt.setp(lines, color="#999999", alpha=0.5)

        return lines_eval, ax

    def plot(self, fig=None):
        """ Plot the distribution updates, the rollouts, the exploration curve and learning curve in
        one figure.

        @param fig:  The figure to plot in (default: None, then a new figure is initialized)
        @return: The figure handle
        """
        if not fig:
            fig = plt.figure(figsize=(20, 5))
        axs = [fig.add_subplot(141 + sp) for sp in range(4)]
        self.plot_distribution_updates(axs[0])
        self.plot_rollouts(axs[1])
        self.plot_exploration_curve(axs[2])
        self.plot_learning_curve(axs[3])
        return fig
