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


import argparse
import copy
import inspect
import os
import warnings
from glob import glob

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
from pylab import mean

from dmpbbo.DmpBboJSONEncoder import saveToJSON

# Avoid warnings when plotting very narrow covariance matrices
warnings.simplefilter("ignore", np.ComplexWarning)


def plotUpdateLines(n_samples_per_update, ax):
    """ Plot vertical lines when a parameter update occurred during the optimization.
    
        Args:
            n_samples_per_update Vector specifying how many samples were used between updates.
            ax Axis object to plot the lines in.
    """

    # Find good number of horizontal update lines to plot
    updates = np.arange(0, len(n_samples_per_update))
    while len(n_samples_per_update) > 20:
        n_samples_per_update = n_samples_per_update[0:-1:5]
        updates = updates[0:-1:5]

    y_limits = ax.get_ylim()

    n = n_samples_per_update
    h = ax.plot([n, n], y_limits)
    plt.setp(h, color="#bbbbbb", linewidth=0.5, zorder=0)

    for ii in range(len(n_samples_per_update) - 1):
        y = y_limits[0] + 0.9 * (y_limits[1] - y_limits[0])
        ax.text(
            n_samples_per_update[ii + 1],
            y,
            str(updates[ii + 1]),
            horizontalalignment="center",
            verticalalignment="top",
            rotation="vertical",
        )

    y = y_limits[0] + 0.95 * (y_limits[1] - y_limits[0])
    ax.text(
        mean(ax.get_xlim()),
        y,
        "number of updates",
        horizontalalignment="center",
        verticalalignment="top",
    )
    ax.set_ylim(y_limits)


def plotLearningCurve(learning_curve, **kwargs):
    """ Plot a learning curve.
    
        Args:
            learning_curve A learning curve that has the following format
        #rows is number of optimization updates
        column 0: Number of samples at which the cost was evaluated
        column 1: The total cost 
        column 2...: Individual cost components (column 1 is their sum)
    
        Args:
            ax Axis to plot the learning curve on.
    
        Args:
            costs_all Vector of costs of each sample (default=[])
    
        Args:
            cost_labels Vector of strings for the different cost components  (default=[]).
    """

    cost_labels = kwargs.get("cost_labels", None)
    ax = kwargs.get("ax") or plt.axes()

    # Plot costs at evaluations
    learning_curve = np.array(learning_curve)
    samples_eval = learning_curve[:, 0]
    costs_eval = learning_curve[:, 1:]
    # Sum of cost components
    lines = ax.plot(samples_eval, costs_eval[:, 0], "-", color="black", linewidth=2)
    # Individual cost components
    if costs_eval.shape[1] > 1:
        lines.extend(ax.plot(samples_eval, costs_eval[:, 1:], "-", linewidth=1))

    # Annotation
    ax.set_xlabel("number of evaluations")
    ax.set_ylabel("cost")
    ax.set_title("Learning curve")

    if cost_labels:
        cost_labels.insert(0, "total cost")
        plt.legend(lines, cost_labels)

    plotUpdateLines(samples_eval, ax)

    return lines, ax


def plotExplorationCurve(exploration_curve, **kwargs):
    """ Plot an exploration curve.
    
        Args:
            exploration An exploration curve that has the following format
        #rows is number of optimization updates
        column 0: Number of samples at which the cost was evaluated
        column 1: The exploration at that update 
    
        Args:
            ax Axis to plot the learning curve on.
    """
    ax = kwargs.get("ax") or plt.axes()

    exploration_curve = np.array(exploration_curve)
    samples_eval = exploration_curve[:, 0]
    explo = exploration_curve[:, 1]

    line = ax.plot(samples_eval, explo, "-", color="green", linewidth=2)
    plotUpdateLines(samples_eval, ax)
    ax.set_xlabel("number of evaluations")
    ax.set_ylabel("sqrt of max. eigen-value of covar")
    ax.set_title("Exploration magnitude")

    return line, ax


def setColor(handle, i_update, n_updates):
    """ Set the color of an object, according to how far the optimization has proceeded.
    
        Args:
            handle: Handle to the object
            i_update: Which update is the optimization at now?
            n_updates: Which is the number of updates the optimization will run?
    """
    if i_update == 0:
        plt.setp(handle, color="r", linewidth=2)
    else:
        c = 1.0 * i_update / n_updates
        cur_color = [0.0 - 0.0 * c, 0.0 + 1.0 * c, 0.0 - 0.0 * c]
        plt.setp(handle, color=cur_color)


def plotUpdate(
    distribution, cost_eval, samples, costs, weights, distribution_new, **kwargs
):
    """ Save an optimization update to a directory.
    
        Args:
            distribution: Gaussian distribution before the update
            cost_eval: Cost of the mean of the distribution
            samples: The samples in the search space
            costs: The cost of each sample
            weights: The weight of each sample
            distribution_new: Gaussian distribution after the update
            kwargs: Can be the following:
                ax: Axis to plot the update on.
                highlight: Whether to highlight this update (default=False)
                plot_samples: Whether to plot the individual samples (default=False)
    """
    ax = kwargs.get("ax") or plt.axes()
    highlight = kwargs.get("highlight", False)
    plot_samples = kwargs.get("plot_samples", False)

    if samples is None:
        plot_samples = False

    n_dims = len(distribution.mean)
    if n_dims == 1:
        raise ValueError(
            "Sorry, only know how to plot for n_dims==2, but you provided n_dims==1"
        )

    # ZZZ Take into consideration block_covar_sizes to plot sub blocks in
    # different subplots

    if n_dims >= 2:
        distr_mean = distribution.mean[0:2]
        # distr_covar = distribution.covar[0:2, 0:2]
        distr_new_mean = distribution_new.mean[0:2]
        # distr_new_covar = distribution_new.covar[0:2, 0:2]
        if samples is not None:
            samples = samples[:, 0:2]

        if plot_samples:
            max_marker_size = 80
            for ii in range(len(weights)):
                cur_marker_size = max_marker_size * weights[ii]
                sample_handle = ax.plot(
                    samples[ii, 0], samples[ii, 1], "o", color="green"
                )
                plt.setp(
                    sample_handle,
                    markersize=cur_marker_size,
                    markerfacecolor=(0.5, 0.8, 0.5),
                    markeredgecolor="none",
                )

                ax.plot(samples[:, 0], samples[:, 1], ".", color="black")
            ax.plot(
                (distr_mean[0], distr_new_mean[0]),
                (distr_mean[1], distr_new_mean[1]),
                "-",
                color="blue",
            )

        mean_handle = ax.plot(distr_mean[0], distr_mean[1], "o", label="old")
        mean_handle_new = ax.plot(
            distr_new_mean[0], distr_new_mean[1], "o", label="new"
        )
        mean_handle_link = ax.plot(
            [distr_mean[0], distr_new_mean[0]], [distr_mean[1], distr_new_mean[1]], "-"
        )
        patch, _ = distribution.plot(ax)
        patch_new, _ = distribution_new.plot(ax)
        # patch = plot_error_ellipse(distr_mean, distr_covar, ax)
        # patch_new = plot_error_ellipse(distr_new_mean, distr_new_covar, ax)
        if highlight:
            plt.setp(mean_handle, color="red")
            plt.setp(mean_handle_new, color="blue")
            plt.setp(patch, edgecolor="red")
            plt.setp(patch_new, edgecolor="blue")
        else:
            plt.setp(mean_handle, color="gray")
            plt.setp(mean_handle_new, color="gray")
            plt.setp(patch, edgecolor="gray")
            plt.setp(patch_new, edgecolor="gray")
        plt.setp(mean_handle_link, color="gray")
    ax.set_aspect("equal")

    ax.set_xlabel("dim 1 (of " + str(n_dims) + ")")
    ax.set_ylabel("dim 2 (of " + str(n_dims) + ")")
    ax.set_title("Search space")
    return mean_handle, mean_handle_new, patch, patch_new


class LearningSession:
    def __init__(self, n_samples_per_update, root_directory=None, **kwargs):

        self._n_samples_per_update = n_samples_per_update
        self._root_dir = root_directory
        self._cache = {}
        self._last_update_added = None

        self.tell(n_samples_per_update, "n_samples_per_update")

        for name, value in kwargs.items():
            self.tell(value, name)

        if root_directory and "cost_function" in kwargs:
            cost_function = kwargs["cost_function"]
            src = inspect.getsourcelines(cost_function.__class__)
            src = " ".join(src[0])
            src = src.replace("(CostFunction)", "")
            filename = os.path.join(root_directory, "cost_function.py")
            with open(filename, "w") as f:
                f.write(src)

    @classmethod
    def from_dir(cls, directory):
        filename = os.path.join(directory, "n_samples_per_update.txt")
        n_samples_per_update = np.atleast_1d(np.loadtxt(filename))
        n_samples_per_update = int(n_samples_per_update[0])
        return cls(n_samples_per_update, directory)

    def addEval(self, i_update, eval_sample, eval_cost):
        self.tell(eval_cost, "eval_cost", i_update)
        self.tell(eval_sample, "eval_sample", i_update)

    def addUpdate(
        self, i_update, distribution, samples, costs, weights, distribution_new=None
    ):
        self.tell(distribution, "distribution", i_update)
        self.tell(samples, "samples", i_update)
        self.tell(costs, "costs", i_update)
        self.tell(weights, "weights", i_update)
        if distribution_new:
            self.tell(distribution_new, "distribution_new", i_update)
        self._last_update_added = i_update

    def getNUpdates(self):
        if not self._root_dir:
            return self._last_update_added
        else:
            update_dirs = sorted(glob(os.path.join(self._root_dir, "update*/")))
            last_dir = update_dirs[-1]
            last_dir = last_dir.replace("update", "")
            last_dir = last_dir.replace(self._root_dir, "")
            last_dir = last_dir.replace("/", "")
            return int(last_dir)

    @staticmethod
    def getBaseName(name, i_update=None, i_sample=None):

        if isinstance(i_sample, int):
            basename = f"{i_sample:03d}_{name}"  # e.g. 0 => "000_name"
        elif i_sample:
            basename = f"{i_sample}_{name}"  # e.g. "eval" => "eval_name"
        else:
            basename = name

        if isinstance(i_update, int):
            # e.g. "update00023/000_name"
            basename = os.path.join(f"update{i_update:05d}", basename)

        return basename

    def exists(self, name, i_update=None, i_sample=None):
        try:
            self.ask(name, i_update, i_sample)
        except Exception:
            return False
        return True

    def ask(self, name, i_update=None, i_sample=None):

        basename = self.getBaseName(name, i_update, i_sample)

        if basename in self._cache:
            return copy.deepcopy(self._cache[basename])

        if not self._root_dir:
            # Not in cache, and no directory given: Error
            raise KeyError("Could not find basename in cache: " + basename)

        else:
            abs_basename = os.path.join(self._root_dir, basename)
            if os.path.isfile(abs_basename + ".json"):
                with open(abs_basename + ".json", "r") as f:
                    obj = jsonpickle.decode(f.read())

            elif os.path.isfile(abs_basename + ".txt"):
                obj = np.loadtxt(abs_basename + ".txt")

            else:
                raise IOError("Could not find file with basename: " + abs_basename)

            self._cache[basename] = obj

            return obj

    def tell(self, obj, name, i_update=None, i_sample=None):

        basename = self.getBaseName(name, i_update, i_sample)

        self._cache[basename] = copy.deepcopy(obj)

        if self._root_dir:
            abs_basename = os.path.join(self._root_dir, basename)

            # Make sure directory exists
            os.makedirs(os.path.dirname(abs_basename), exist_ok=True)

            if isinstance(obj, list) or isinstance(obj, int):
                # Try to convert list to numpy array
                obj = np.atleast_1d(np.array(obj))

            if isinstance(obj, np.ndarray):
                filename = abs_basename + ".txt"
                np.savetxt(filename, obj)
            else:
                filename = abs_basename + ".json"
                saveToJSON(obj, filename)

            return filename

    def getEvalCosts(self, i_update):
        has_eval = self.exists("cost", 0, "eval")
        if has_eval:
            cost_eval = self.ask("cost", i_update, "eval")
            # Evaluation at the beginning of an update
            i_sample = i_update * self._n_samples_per_update
        else:
            update_costs = self.getSampleCosts(i_update)
            cost_eval = np.mean(update_costs, axis=0)
            # Evaluation between two updates (mean of all samples)
            i_sample = (i_update + 0.5) * self._n_samples_per_update
        return cost_eval, i_sample

    def getSampleCosts(self, i_update):
        if self.exists("costs", i_update):
            # Everything in one file
            sample_costs = self.ask("costs", i_update)
        else:
            # One file for each sample
            n = self._n_samples_per_update
            sample_costs = [self.ask("costs", i_update, s) for s in range(n)]
        return np.array(sample_costs)

    def getLearningCurve(self):
        learning_curve = []
        n_updates = self.getNUpdates()
        for i_update in range(n_updates):
            costs_eval, i_sample = self.getEvalCosts(i_update)
            costs_eval = np.atleast_1d(costs_eval)
            learning_curve.append(np.concatenate(([i_sample], costs_eval)))

        cost_labels = []
        if self.exists("cost_function"):
            cost_function = self.ask("cost_function")
            cost_labels = cost_function.costLabels()
        elif self.exists("task"):
            cost_function = self.ask("task")
            cost_labels = cost_function.costLabels()

        return learning_curve, cost_labels

    def plotLearningCurve(self, ax=None):
        """
        has eval or not
        has cost components or not
        saved in costs or 000_cost...
        """
        if not ax:
            ax = plt.axes()

        # Plot costs of individual samples
        costs = []
        for i_update in range(self.getNUpdates()):
            sample_costs = self.getSampleCosts(i_update)
            costs.extend([c[0] for c in sample_costs])  # Only include sum
        ax.plot(costs, ".", color="gray")

        # Plot learning curve itself
        learning_curve, cost_labels = self.getLearningCurve()
        return plotLearningCurve(learning_curve, ax=ax, cost_labels=cost_labels)

    def plotExplorationCurve(self, ax=None):
        n_updates = self.getNUpdates()
        curve = np.zeros((n_updates, 2))
        for i_update in range(n_updates):
            distribution = self.ask("distribution", i_update)
            cur_exploration = np.sqrt(distribution.maxEigenValue())
            curve[i_update, 0] = i_update * self._n_samples_per_update
            curve[i_update, 1] = cur_exploration

        if not ax:
            ax = plt.axes()
        return plotExplorationCurve(curve, ax=ax)

    def plotDistributionUpdates(self, ax=None):
        if not ax:
            ax = plt.axes()

        n_updates = self.getNUpdates()
        for i_update in range(n_updates):
            highlight = i_update == 0
            self.plotDistributionUpdate(i_update, ax, highlight=highlight)
        return ax  # zzz

    def plotDistributionUpdate(self, i_update, ax=None, **kwargs):
        # ax = kwargs.get("ax") or plt.axes()
        highlight = kwargs.get("highlight", False)
        plot_samples = kwargs.get("plot_samples", False)

        distribution = self.ask("distribution", i_update)
        costs_eval = self.getEvalCosts(i_update)
        costs = self.getSampleCosts(i_update)
        samples = self.ask("samples", i_update)
        weights = self.ask("weights", i_update)
        distribution_new = self.ask("distribution_new", i_update)

        if not ax:
            ax = plt.axes()

        return plotUpdate(
            distribution,
            costs_eval,
            samples,
            costs,
            weights,
            distribution_new,
            ax=ax,
            highlight=highlight,
            plot_samples=plot_samples,
        )

    def plot(self, fig=None):
        if not fig:
            fig = plt.figure(figsize=(15, 5))
        axs = [fig.add_subplot(131 + sp) for sp in range(3)]
        self.plotDistributionUpdates(axs[0])
        self.plotExplorationCurve(axs[1])
        self.plotLearningCurve(axs[2])
        return fig

    @staticmethod
    def plotMultiple(sessions):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory", type=str, help="directory from which to read the learning session"
    )
    args = parser.parse_args()

    session = LearningSession.from_dir(args.directory)
    session.plot()
    plt.show()
