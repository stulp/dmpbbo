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
import inspect
from pathlib import Path

from matplotlib import pyplot as plt

from dmpbbo.bbo.LearningSession import LearningSession
import dmpbbo.json_for_cpp as jc


class LearningSessionTask(LearningSession):
    def __init__(self, n_samples_per_update, directory=None, **kwargs):
        super().__init__(n_samples_per_update, directory, **kwargs)
        self.task_ = kwargs.get("task", None)
        self.task_solver_ = kwargs.get("task_solver", None)

        if directory and self.task_:
            src = inspect.getsourcelines(self.task_.__class__)
            src = " ".join(src[0])
            src = src.replace("(Task)", "")
            filename = Path(directory, "task.py")
            with open(filename, "w") as f:
                f.write(src)

    def tell(self, obj, name, i_update=None, i_sample=None):
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
        self.tell(sample, "sample", i_update, i_sample)
        self.tell(cost_vars, "cost_vars", i_update, i_sample)
        self.tell(cost, "cost", i_update, i_sample)

    def add_eval_task(self, i_update, eval_sample, eval_cost_vars, eval_cost):
        super().add_eval(i_update, eval_sample, eval_cost)
        self.tell(eval_cost_vars, "eval_cost_vars", i_update)

    @staticmethod
    def _set_style(handle, i_update, n_updates):
        """ Set the color of an object, according to how far the optimization has proceeded.
        
            Args:
                handle: Handle to the object
                i_update: Which update is the optimization at now?
                n_updates: Which is the number of updates the optimization will run?
        """
        if i_update == 0:
            plt.setp(handle, color=[0.8, 0.0, 0.0], linewidth=2)
        else:
            c = 1.0 * i_update / n_updates
            cur_color = [0.3 - 0.3 * c, 0.0 + 1.0 * c, 0.0 - 0.0 * c]
            plt.setp(handle, color=cur_color, linewidth=2)

    def plot_rollouts(self, ax=None):
        if not ax:
            ax = plt.axes()
        all_lines = []
        n_updates = self.get_n_updates()
        for i_update in range(n_updates):
            lines, _ = self.plot_rollouts_update(i_update, ax, True, False)
            LearningSessionTask._set_style(lines, i_update, n_updates)
            all_lines.extend(lines)
        return all_lines, ax

    def plot_rollouts_update(self, i_update, ax=None, plot_eval=True, plot_samples=False):
        if not ax:
            ax = plt.axes()

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
        if not fig:
            fig = plt.figure(figsize=(20, 5))
        axs = [fig.add_subplot(141 + sp) for sp in range(4)]
        self.plot_distribution_updates(axs[0])
        self.plot_rollouts(axs[1])
        self.plot_exploration_curve(axs[2])
        self.plot_learning_curve(axs[3])
        return fig
