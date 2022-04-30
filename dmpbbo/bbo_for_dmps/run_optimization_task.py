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


from dmpbbo.bbo_for_dmps.LearningSessionTask import LearningSessionTask


def run_optimization_task(
    task,
    task_solver,
    initial_distribution,
    updater,
    n_updates,
    n_samples_per_update,
    directory=None,
):

    session = LearningSessionTask(
        n_samples_per_update, directory, task=task, task_solver=task_solver, updater=updater
    )

    distribution = initial_distribution

    # Optimization loop
    for i_update in range(n_updates):
        print(f"Update: {i_update}")

        # 0. Get cost of current distribution mean
        cost_vars_eval = task_solver.perform_rollout(distribution.mean)
        cost_eval = task.evaluate_rollout(cost_vars_eval, distribution.mean)

        # Bookkeeping
        session.add_eval_task(i_update, distribution.mean, cost_vars_eval, cost_eval)

        # 1. Sample from distribution
        samples = distribution.generate_samples(n_samples_per_update)

        # 2. Evaluate the samples
        costs = []
        for i_sample, sample in enumerate(samples):

            # 2A. Perform the rollout
            cost_vars = task_solver.perform_rollout(sample)

            # 2B. Evaluate the rollout
            cur_cost = task.evaluate_rollout(cost_vars, sample)
            costs.append(cur_cost)

            # Bookkeeping
            session.add_rollout(i_update, i_sample, sample, cost_vars, cur_cost)

        # 3. Update parameters
        distribution_new, weights = updater.update_distribution(distribution, samples, costs)

        # Bookkeeping
        session.add_update(i_update, distribution, samples, costs, weights, distribution_new)

        distribution = distribution_new

    return session
