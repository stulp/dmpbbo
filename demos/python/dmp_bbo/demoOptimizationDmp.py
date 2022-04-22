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


import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add relative path, in case PYTHONPATH is not set
lib_path = os.path.abspath("../../python/")
sys.path.append(lib_path)

from functionapproximators.FunctionApproximatorLWR import *
from functionapproximators.FunctionApproximatorRBFN import *

from bbo.DistributionGaussian import DistributionGaussian
from bbo.updaters import *

from dmp.Dmp import Dmp

from dmp_bbo.Task import Task
from dmp_bbo.TaskSolver import TaskSolver
from dmp_bbo.TaskSolverDmp import TaskSolverDmp
from dmp_bbo.runOptimizationTask import runOptimizationTask

from dmp_bbo.tasks.TaskViapoint import TaskViapoint


if __name__ == "__main__":

    directory = None
    if len(sys.argv) > 1:
        directory = sys.argv[1]

    for n_dims in [1, 2]:

        # Some DMP parameters
        tau = 0.5
        y_init = np.linspace(1.8, 2.0, n_dims)
        y_attr = np.linspace(4.0, 3.0, n_dims)

        # initialize function approximators with random values
        function_apps = []
        intersection_height = 0.9
        for n_basis in [8, 9]:

            fa = FunctionApproximatorRBFN(n_basis, intersection_height)
            fa.train(np.linspace(0, 1, 100), np.zeros(100))

            fa.setSelectedParamNames("weights")
            random_weights = 0 * np.random.normal(0, 1, n_basis)
            fa.setParamVector(random_weights)

            function_apps.append(fa)

        # Initialize Dmp
        dmp = Dmp(tau, y_init, y_attr, function_apps)
        dmp.setSelectedParamNames("weights")
        # dmp.setSelectedParamNames(['goal','weights'])

        # Make the task
        viapoint = 3 * np.ones(n_dims)
        viapoint_time = 0.3
        if n_dims == 2:
            # Do not pass through viapoint at a specific time, but rather pass
            # through it at any time.
            viapoint_time = None
        viapoint_radius = 0.1
        goal = y_attr
        goal_time = 1.1 * tau
        viapoint_weight = 1.0
        acceleration_weight = 0.0001
        goal_weight = 0.0
        task = TaskViapoint(
            viapoint,
            viapoint_time,
            viapoint_radius,
            goal,
            goal_time,
            viapoint_weight,
            acceleration_weight,
            goal_weight,
        )

        # Make task solver, based on a Dmp
        dt = 0.01
        integrate_dmp_beyond_tau_factor = 1.5
        task_solver = TaskSolverDmp(dmp, dt, integrate_dmp_beyond_tau_factor)

        n_search = dmp.getParamVectorSize()

        covar_update = "cma"
        mean_init = np.full(n_search, 0.0)
        covar_init = 1000.0 * np.eye(n_search)
        distribution = DistributionGaussian(mean_init, covar_init)

        eliteness = 10
        weighting_method = "PI-BB"  # or 'CEM' or 'CMA-ES'
        if covar_update == "none":
            updater = UpdaterMean(eliteness, weighting_method)
        elif covar_update == "decay":
            covar_decay_factor = 0.9
            updater = UpdaterCovarDecay(eliteness, weighting_method, covar_decay_factor)
        else:
            min_level = 0.000001
            max_level = None
            diag_only = False
            learning_rate = 0.5
            updater = UpdaterCovarAdaptation(
                eliteness,
                weighting_method,
                max_level,
                min_level,
                diag_only,
                learning_rate,
            )

        n_samples_per_update = 10
        n_updates = 40

        fig = plt.figure(n_dims, figsize=(15, 5))
        fig.canvas.set_window_title("Optimization with covar_update=" + covar_update)

        learning_curve = runOptimizationTask(
            task,
            task_solver,
            distribution,
            updater,
            n_updates,
            n_samples_per_update,
            fig,
            directory,
        )

    plt.show()
