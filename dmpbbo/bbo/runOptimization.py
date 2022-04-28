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


from dmpbbo.bbo.LearningSession import LearningSession


def run_optimization(
    cost_function,
    initial_distribution,
    updater,
    n_updates,
    n_samples_per_update,
    directory=None,
):
    """ Run an evolutionary optimization process, see \ref page_bbo
    
        Args:
            cost_function: The cost function to optimize
            initial_distribution: The initial parameter distribution
            updater: The Updater used to update the parameters
            n_updates: The number of updates to perform
            n_samples_per_update: The number of samples per update
            directory: Optional directory to save to (default: don't save)
    Returns:
            A learning curve that has the following format
        #rows is number of optimization updates
        column 0: Number of samples at which the cost was evaluated
        column 1: The total cost 
        column 2...: Individual cost components (column 1 is their sum)
    """

    session = LearningSession(
        n_samples_per_update, directory, cost_function=cost_function, updater=updater
    )

    distribution = initial_distribution

    # Optimization loop
    for i_update in range(n_updates):

        # 0. Get cost of current distribution mean
        cost_eval = cost_function.evaluate(distribution.mean)

        # 1. Sample from distribution
        samples = distribution.generate_samples(n_samples_per_update)

        # 2. Evaluate the samples
        costs = [cost_function.evaluate(sample) for sample in samples]

        # 3. Update the distribution
        distribution_new, weights = updater.update_distribution(
            distribution, samples, costs
        )

        # Bookkeeping
        session.add_eval(i_update, distribution.mean, cost_eval)
        session.add_update(
            i_update, distribution, samples, costs, weights, distribution_new
        )

        distribution = distribution_new

    return session
