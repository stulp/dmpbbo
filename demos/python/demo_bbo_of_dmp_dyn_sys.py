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
""" Script for training a DMP from a trajectory. """

import numpy as np

import dmpbbo.json_for_cpp as json_for_cpp
from dmpbbo.bbo.DistributionGaussian import DistributionGaussian
from dmpbbo.bbo.updaters import UpdaterCovarDecay
from dmpbbo.bbo_of_dmps.Task import Task
from dmpbbo.bbo_of_dmps.TaskSolver import TaskSolver
from dmpbbo.bbo_of_dmps.run_optimization_task import run_optimization_task
from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.dynamicalsystems.SigmoidSystem import SigmoidSystem
from dmpbbo.dynamicalsystems.SpringDamperSystem import SpringDamperSystem
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN
from dmpbbo.dynamicalsystems.ExponentialSystem import ExponentialSystem

from matplotlib import pyplot as plt

class TaskFitTrajectory(Task):
    """ Task in which a trajectory has to pass through a viapoint."""

    def __init__(self,  traj_demonstrated, **kwargs):
        self.traj_demonstrated = traj_demonstrated
        self.i_diff = 3

    def get_cost_labels(self):
        return ["mean(abs(diff))", "abs(goaldiff)"]

    def evaluate_rollout(self, cost_vars, sample):
        """The cost function which defines the task.

        @param cost_vars: All the variables relevant to computing the cost. These are determined by
            TaskSolver.perform_rollout(). For further information see the tutorial on "bbo_of_dmps".
        @param sample: The sample from which the rollout was generated. Passing this to the cost
            function is useful when performing regularization on the sample.
        @return: costs The scalar cost components for the sample. The first item costs[0] should
            contain the total cost.
        """
        cost_vars_demo = self.traj_demonstrated.as_matrix()
        diff = np.mean(np.abs(cost_vars_demo[:, self.i_diff] - cost_vars[:, self.i_diff]))
        diff_goal = 5*np.mean(np.abs(cost_vars_demo[-25:-1, 1] - cost_vars[-25:-1,
                                                                      1]))
        return [diff+diff_goal, diff, diff_goal]

    def plot_rollout(self, cost_vars, ax=None):
        """ Plot a rollout (the cost-relevant variables).

        @param cost_vars: Rollout to plot
        @param ax: Axis to plot on (default: None, then a new axis a created)
        @return: line handles and axis
        """

        if not ax:
            ax = plt.axes()

        ts = cost_vars[:, 0]
        cost_vars_demo = self.traj_demonstrated.as_matrix()

        lines_dem = ax.plot(ts,cost_vars_demo[:,self.i_diff],label="demonstration")
        plt.setp(lines_dem, linestyle="-", linewidth=4, color=(0.8, 0.8, 0.8))
        lines_rep = ax.plot(ts,cost_vars[:, self.i_diff],label="reproduced")
        plt.setp(lines_rep, linestyle="-", linewidth=2, color=(0.0, 0.0, 0.5))

        return lines_rep, ax


class TaskSolverDmpDynSys(TaskSolver):

    def __init__(self, trajectory):
        self._trajectory = trajectory

    def get_dmp(self, sample, function_apps=None):
        y_init = self._trajectory.ys[0, :]
        y_attr = self._trajectory.ys[-1, :]

        tau = sample[0]

        damping = sample[1]
        constant = sample[2]
        mass = sample[3]
        transf_system = SpringDamperSystem(tau, y_init, y_attr, damping, constant, mass)

        tau_goal = sample[4]
        param1 = sample[5]
        goal_system = ExponentialSystem(tau_goal, y_init, y_attr, param1)

        dmp = Dmp.from_traj(self._trajectory, function_apps, dmp_type="KULVICIUS_2012_JOINING",
            goal_system=goal_system, transformation_system=transf_system
        )
        return dmp


    def perform_rollout(self, sample, **kwargs):
        dmp = self.get_dmp(sample)
        xs, xds, _, _ = dmp.analytical_solution()
        traj = dmp.states_as_trajectory(dmp.ts_train, xs, xds)

        cost_vars = traj.as_matrix()
        return cost_vars


class TaskFitTargets(Task):
    """ Task in which a trajectory has to pass through a viapoint."""

    def __init__(self,  inputs, targets, **kwargs):
        self.inputs = inputs
        self.targets = targets

    def get_cost_labels(self):
        return ["mean(abs(diff))"]

    def evaluate_rollout(self, cost_vars, sample):
        diff = np.mean(np.square(self.targets - cost_vars[:, 0]))
        return [diff]

    def plot_rollout(self, cost_vars, ax=None):
        if not ax:
            ax = plt.axes()

        lines_dem = ax.plot(self.inputs,self.targets,label="targets")
        plt.setp(lines_dem, linestyle="-", linewidth=4, color=(0.8, 0.8, 0.8))
        lines_rep = ax.plot(self.inputs,cost_vars[:, 0],label="predictions")
        plt.setp(lines_rep, linestyle="-", linewidth=2, color=(0.0, 0.0, 0.5))

        return lines_rep, ax

class TaskSolverFunctionApp(TaskSolver):

    def __init__(self, fa, inputs):
        self.fa = fa
        self.inputs = inputs

    def perform_rollout(self, sample, **kwargs):
        self.fa.set_param_vector(sample)
        predictions = self.fa.predict(self.inputs)
        predictions = predictions.reshape((-1, 1))
        return predictions




def main():
    """ Main function for script. """

    # Train a DMP with a trajectory
    y_first = np.array([0.0])
    y_last = np.array([1.0])
    dim_dmp = len(y_first)
    traj_demo = Trajectory.from_min_jerk(np.linspace(0, 1.0, 101), y_first, y_last)
    traj_end =  Trajectory.from_min_jerk(np.linspace(0, 0.25, 26), y_last, y_last)
    traj_demo.append(traj_end)
    tau = 1.0
    ts = traj_demo.ts
    #traj_demo = Trajectory.loadtxt("trajectory.txt")

    task = TaskFitTrajectory(traj_demo)
    task_solver_dyn_sys = TaskSolverDmpDynSys(traj_demo)

    optimize_first = False
    if optimize_first:

        damping = 20
        mean_init = [
            tau, # tau = sample[0]
            damping, # damping = sample[1]
            damping*damping/4, # constant = sample[2]
            1.0,  # mass = sample[3]
            tau, # tau_goal = sample[4]
            5 # param1 = sample[5]
        ]
        print(mean_init)
        covar_init = np.diag([0.01, 1.0, 10.0, 0.1, 0.01, 0.5])
        covar_init = np.diag([0.0, 1.0, 10.0, 0.1, 0.1, 0.5])
        distribution = DistributionGaussian(mean_init, covar_init)
        updater = UpdaterCovarDecay(eliteness=10, weighting_method="PI-BB", covar_decay_factor=0.9)

        n_samples_per_update = 20
        n_updates = 50

        session = run_optimization_task(
            task, task_solver_dyn_sys, distribution, updater, n_updates, n_samples_per_update
        )

        distribution_opt = session.ask('distribution', n_updates-1)
        mean_opt = distribution_opt.mean
        print(mean_opt)

        axs_dmp = None
        axs_traj = None
        for i_update in [0, n_updates-1]:
            distribution = session.ask('distribution', i_update)
            dmp = task_solver_dyn_sys.get_dmp(distribution.mean)
            xs, xds, forcing_terms, fa_outputs = dmp.analytical_solution()
            _, axs_dmp = dmp.plot(ts, xs, xds, forcing_terms=forcing_terms, fa_outputs=fa_outputs,
                                  axs=axs_dmp)
            lhs, axs_traj = dmp.plot_comparison(traj_demo,axs=axs_traj)

            if i_update == 0:
                plt.setp(lhs[3:], linestyle="-", linewidth=2, color=(0.6, 0.0, 0.0))
            else:
                plt.setp(lhs[3:], linestyle="-", linewidth=2, color=(0.0, 0.6, 0.0))
            plt.legend(['demonstrated', 'reproduced (default)', None, 'reproduced (optimized)'])

        fig = session.plot()

    else:
        mean_opt = [1.0, 17.65977591,  98.21818331,  2.42016606,  1.15073151,  4.20203766]

    n_plots = 2
    fig = plt.figure(figsize=(5 * n_plots, 4))
    axs = [fig.add_subplot(1, n_plots, i + 1) for i in range(n_plots)]
    for use_optimized in [False, True]:
        if use_optimized:
            mean_opt = [1.0, 17.65977591, 98.21818331, 2.42016606, 1.15073151, 4.20203766]
            centers = np.array([0.1, 0.42, 0.7, 1.0])
            widths = np.array([0.07, 0.12, 0.07, 0.1])
        else:
            mean_opt = [1.0, 20, 100.0, 1.0, 1.0, 5]
            centers = np.array([0.13, 0.5, 0.75, 1.0])
            widths = np.array([0.10, 0.11, 0.03, 0.1])

        dmp = task_solver_dyn_sys.get_dmp(mean_opt)
        fa_inputs_phase, fa_targets = dmp._compute_targets(traj_demo)

        n_basis_functions = 4
        fa = FunctionApproximatorRBFN(n_basis_functions, 0.5)
        fa._meta_params.update({"centers": centers, "widths": widths})
        fa.train(fa_inputs_phase, fa_targets)
        ax =  axs[1] if use_optimized else axs[0]
        fa.plot(fa_inputs_phase, targets=fa_targets, plot_model_parameters=True, ax=ax)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([-25, 20])
        ax.set_xlabel('phase')
        ax.set_ylabel('ydd')
        ax.set_title('dyn_sys optimized' if use_optimized else 'dyn_sys default')
    plt.show()


    a = 1/0

    names = ["weights", "centers", "widths"]
    fa.set_selected_param_names(names)

    task = TaskFitTargets(fa_inputs_phase, fa_targets)
    task_solver_function_app = TaskSolverFunctionApp(fa, fa_inputs_phase)
    mean_init = fa.get_param_vector()
    print(mean_init)
    covar_diag = []
    if 'weights' in names:
        covar_diag.extend([0.0]*n_basis_functions)
    if 'centers' in names:
        covar_diag.extend([0.001]*n_basis_functions)
    if 'widths' in names:
        covar_diag.extend([0.001]*n_basis_functions)
    print(covar_diag)
    covar_init = np.diag(covar_diag)
    distribution = DistributionGaussian(mean_init, covar_init)
    updater = UpdaterCovarDecay(eliteness=10, weighting_method="PI-BB", covar_decay_factor=0.8)

    n_samples_per_update = 20
    n_updates = 30

    session = run_optimization_task(
        task, task_solver_function_app, distribution, updater, n_updates, n_samples_per_update
    )
    fig = session.plot()

    plt.show()

if __name__ == "__main__":
    main()
