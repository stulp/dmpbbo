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


import os
import sys
import numpy as np
import matplotlib.pyplot as plt


from dmpbbo.functionapproximators.FunctionApproximatorLWR import *
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import *

from dmpbbo.bbo.DistributionGaussian import DistributionGaussian
from dmpbbo.bbo.updaters import *

from dmpbbo.dmp.Dmp import Dmp

from dmpbbo.dmp_bbo.Task import Task
from dmpbbo.dmp_bbo.TaskSolver import TaskSolver
from dmpbbo.dmp_bbo.TaskSolverDmp import TaskSolverDmp
from dmpbbo.dmp_bbo.runOptimizationTask import runOptimizationTask

from TaskViapoint import TaskViapoint


def runDemo(directory, n_dims):

    # Some DMP parameters
    tau = 0.5
    y_init = np.linspace(1.8, 2.0, n_dims)
    y_attr = np.linspace(4.0, 3.0, n_dims)

    # initialize function approximators with random values
    function_apps = []
    intersection_height = 0.8
    for n_basis in [6, 7]:

        fa = FunctionApproximatorRBFN(n_basis, intersection_height)
        fa.train(np.linspace(0, 1, 100), np.zeros(100))

        fa.setSelectedParamNames("weights")
        random_weights = 10.0 * np.random.normal(0, 1, n_basis)
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

    mean_init = np.full(n_search, 0.0)
    covar_init = 1000.0 * np.eye(n_search)
    distribution = DistributionGaussian(mean_init, covar_init)

    covar_update = "cma"
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
            eliteness, weighting_method, max_level, min_level, diag_only, learning_rate
        )

    n_samples_per_update = 10
    n_updates = 40

    session = runOptimizationTask(
        task,
        task_solver,
        distribution,
        updater,
        n_updates,
        n_samples_per_update,
        directory,
    )
    fig = session.plot()
    fig.canvas.set_window_title("Optimization with covar_update=" + covar_update)


if __name__ == "__main__":

    directory = sys.argv[1] if len(sys.argv) > 1 else None

    for n_dims in [1, 2]:
        runDemo(directory, n_dims)

    plt.show()
