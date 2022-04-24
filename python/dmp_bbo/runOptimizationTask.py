# This file is part of DmpBbo, a set of libraries and programs for the
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
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


import numpy as np
import math
import os
import sys
import matplotlib.pyplot as plt
from collections import OrderedDict

lib_path = os.path.abspath("../../python/")
sys.path.append(lib_path)


from dmp_bbo.Rollout import Rollout
from dmp_bbo.LearningSessionTask import *


def runOptimizationTask(
    task,
    task_solver,
    initial_distribution,
    updater,
    n_updates,
    n_samples_per_update,
    directory=None,
):

    session = LearningSessionTask(
        n_samples_per_update, directory, task=task, updater=updater
    )

    distribution = initial_distribution

    # Optimization loop
    for i_update in range(n_updates):
        print(f"Update: {i_update}")

        # 0. Get cost of current distribution mean
        cost_vars_eval = task_solver.performRollout(distribution.mean)
        cost_eval = task.evaluateRollout(cost_vars_eval, distribution.mean)

        # Bookkeeping
        session.addEval(i_update, distribution.mean, cost_vars_eval, cost_eval)

        # 1. Sample from distribution
        samples = distribution.generateSamples(n_samples_per_update)

        # 2. Evaluate the samples
        costs = []
        for i_sample, sample in enumerate(samples):

            # 2A. Perform the rollouts
            cost_vars = task_solver.performRollout(sample)

            # 2B. Evaluate the rollouts
            cur_cost = task.evaluateRollout(cost_vars, sample)
            costs.append(cur_cost)

            # Bookkeeping
            session.addRollout(i_update, i_sample, sample, cost_vars, cur_cost)

        # 3. Update parameters
        distribution_new, weights = updater.updateDistribution(
            distribution, samples, costs
        )

        # Bookkeeping
        session.addUpdate(
            i_update, distribution, samples, costs, weights, distribution_new
        )

        distribution = distribution_new

    return session
