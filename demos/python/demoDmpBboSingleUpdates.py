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


import sys

from dmpbbo.bbo.updaters import *
from dmpbbo.dmp.Dmp import Dmp
from dmpbbo.dmp_bbo.run_one_update import *
from dmpbbo.dmp_bbo.TaskSolverDmp import TaskSolverDmp
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import *
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

    eliteness = 10
    weighting_method = "PI-BB"
    covar_decay_factor = 0.9
    updater = UpdaterCovarDecay(eliteness, weighting_method, covar_decay_factor)

    n_samples_per_update = 10
    n_updates = 20

    session = runOptimizationTaskPrepare(
        directory, task, task_solver, distribution, n_samples_per_update, updater, dmp
    )

    for i_update in range(n_updates):
        dmp_eval = session.ask("dmp", i_update, "eval")
        cost_vars_eval = task_solver.performRolloutDmp(dmp_eval)
        session.tell(cost_vars_eval, "cost_vars", i_update, "eval")

        for i_sample in range(n_samples_per_update):
            dmp_sample = session.ask("dmp", i_update, i_sample)
            cost_vars = task_solver.performRolloutDmp(dmp_sample)
            session.tell(cost_vars, "cost_vars", i_update, i_sample)

        runOptimizationTaskOneUpdate(session, i_update)

    return session.plot()


if __name__ == "__main__":

    directory = "/tmp/demoDmpBboSingleUpdates"
    if len(sys.argv) > 1:
        directory = sys.argv[1]

    for n_dims in [1]:
        runDemo(directory, n_dims)

    plt.show()
