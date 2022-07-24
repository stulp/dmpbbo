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
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN
from dmpbbo.dynamicalsystems.ExponentialSystem import ExponentialSystem

from matplotlib import pyplot as plt

class TaskFitTrajectory(Task):
    """ Task in which a trajectory has to pass through a viapoint."""

    def __init__(self,  traj_demonstrated, **kwargs):
        self.traj_demonstrated = traj_demonstrated

    def get_cost_labels(self):
        return ["mean_abs_diff"]

    def evaluate_rollout(self, cost_vars, sample):
        """The cost function which defines the task.

        @param cost_vars: All the variables relevant to computing the cost. These are determined by
            TaskSolver.perform_rollout(). For further information see the tutorial on "bbo_of_dmps".
        @param sample: The sample from which the rollout was generated. Passing this to the cost
            function is useful when performing regularization on the sample.
        @return: costs The scalar cost components for the sample. The first item costs[0] should
            contain the total cost.
        """
        ys_reproduced = cost_vars[:, 1]
        yds_reproduced = cost_vars[:, 2]
        ydds_reproduced = cost_vars[:, 3]

        diff = np.mean(np.abs(self.traj_demonstrated.yds - yds_reproduced))

        return [diff]

    def plot_rollout(self, cost_vars, ax=None):
        """ Plot a rollout (the cost-relevant variables).

        @param cost_vars: Rollout to plot
        @param ax: Axis to plot on (default: None, then a new axis a created)
        @return: line handles and axis
        """

        if not ax:
            ax = plt.axes()

        ts = cost_vars[:, 0]
        ys_reproduced = cost_vars[:, 1]
        yds_reproduced = cost_vars[:, 2]
        ydds_reproduced = cost_vars[:, 3]

        lines_dem = ax.plot(ts,self.traj_demonstrated.yds,label="demonstration")
        plt.setp(lines_dem, linestyle="-", linewidth=4, color=(0.8, 0.8, 0.8))
        lines_rep = ax.plot(ts,yds_reproduced,label="reproduced")
        plt.setp(lines_rep, linestyle="--", linewidth=2, color=(0.0, 0.0, 0.5))

        return lines_rep, ax

class TaskSolverDmpDynSys(TaskSolver):
    """ TaskSolver that integrates a DMP.
    """

    def __init__(self, trajectory):
        self._trajectory = trajectory

    def get_dmp(self, sample, function_apps=None):
        tau = sample[0]
        spring_damper_alpha = sample[1]
        goal_system_param1 = sample[2]
        tau_goal = sample[3]

        y_init = self._trajectory.ys[0, :]
        y_attr = self._trajectory.ys[-1, :]

        goal_system = ExponentialSystem(tau_goal, y_init, y_attr, goal_system_param1)

        #def __init__(self, tau, x_init, max_rate, inflection_ratio):
        #goal_system = SigmoidSystem(tau, y_init, goal_system_param1, goal_system_param2)
        dmp = Dmp.from_traj(self._trajectory, function_apps, dmp_type="KULVICIUS_2012_JOINING",
            goal_system=goal_system, alpha_spring_damper=spring_damper_alpha
        )
        return dmp


    def perform_rollout(self, sample, **kwargs):
        """ Perform rollouts, that is, given a set of samples, determine all the variables that
        are relevant to evaluating the cost function.

        @param sample: The sample to perform the rollout for
        @return: The variables relevant to computing the cost.
        """
        dmp = self.get_dmp(sample)
        ts = self._trajectory.ts
        xs, xds, _, _ = dmp.analytical_solution(ts)
        traj = dmp.states_as_trajectory(ts, xs, xds)

        cost_vars = traj.as_matrix()
        return cost_vars


def main():
    """ Main function for script. """

    # Train a DMP with a trajectory
    tau = 1.0
    ts = np.linspace(0, tau, 101)
    y_first = np.array([0.0])
    y_last = np.array([1.0])
    traj_demo = Trajectory.from_min_jerk(ts, y_first, y_last)
    #traj_demo = Trajectory.loadtxt("trajectory.txt")
    #ts = traj_demo.ts

    task_solver = TaskSolverDmpDynSys(traj_demo)

    mean_init = np.array([tau, 15, 5, tau])
    covar_init = np.diag([0.01, 0.5, 0.5, 0.01])
    distribution = DistributionGaussian(mean_init, covar_init)
    updater = UpdaterCovarDecay(eliteness=10, weighting_method="PI-BB", covar_decay_factor=0.9)

    n_samples_per_update = 10
    n_updates = 10

    task = TaskFitTrajectory(traj_demo)
    session = run_optimization_task(
        task, task_solver, distribution, updater, n_updates, n_samples_per_update
    )

    axs_dmp = None
    axs_traj = None
    for i_update in [0, n_updates-1]:
        distribution = session.ask('distribution', i_update)
        dmp = task_solver.get_dmp(distribution.mean)
        xs, xds, forcing_terms, fa_outputs = dmp.analytical_solution()
        _, axs_dmp = dmp.plot(ts, xs, xds, forcing_terms=forcing_terms, fa_outputs=fa_outputs,
                              axs=axs_dmp)
        _, axs_traj = dmp.plot_comparison(traj_demo,axs=axs_traj)

    fig = session.plot()

    if 1==0:
        function_apps = [FunctionApproximatorRBFN(10, 0.9) for _ in range(traj.dim)]

        goal_system = ExponentialSystem(tau, y_first, y_last, 5)
    
        #dmp = Dmp.from_traj(traj, function_apps, dmp_type="KULVICIUS_2012_JOINING",
        #    goal_system=goal_system, alpha_spring_damper=15.0
        #)

        function_apps = None
        dmp = Dmp.from_traj(traj, function_apps, dmp_type="KULVICIUS_2012_JOINING")
    
        #dmp.set_selected_param_names("weights")
        #v = dmp.get_param_vector()
        #print(v)
        #v.fill(0.0)
        #print(v)
        #dmp.set_param_vector(v)

        # Compute analytical solution
        ts = np.linspace(0, 0.75, 151)
        xs, xds, forcing_terms, fa_outputs = dmp.analytical_solution(ts)
        dmp.plot(ts, xs, xds, forcing_terms=forcing_terms, fa_outputs=fa_outputs)

        lines, axs = traj.plot()
        plt.setp(lines, linestyle="-", linewidth=4, color=(0.8, 0.8, 0.8))
        plt.setp(lines, label="demonstration")
        traj_reproduced = dmp.states_as_trajectory(ts, xs, xds)
        lines, axs = traj_reproduced.plot(axs)
        plt.setp(lines, linestyle="--", linewidth=2, color=(0.0, 0.0, 0.5))
        plt.setp(lines, label="reproduced")



    plt.show()

if __name__ == "__main__":
    main()
