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
"""Script for bbo demo with multiple cost components."""


import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dmpbbo.bbo.CostFunction import CostFunction
from dmpbbo.bbo.DistributionGaussian import DistributionGaussian
from dmpbbo.bbo.run_optimization import run_optimization
from dmpbbo.bbo.updaters import UpdaterCovarDecay


class DemoCostFunctionDistanceToPoint(CostFunction):
    """ CostFunction in which the distance to a pre-defined point must be minimized."""

    def __init__(self, point, regularization_weight=1.0):
        """ Constructor.

        @param point: Point to which distance must be minimized.
        """
        self.point = np.asarray(point)
        self.regularization_weight = regularization_weight

    def evaluate(self, sample):
        """ Evaluate one sample with the cost function.

        @param sample: The sample to evaluate
        @return: distance from sample to point and a regularization term
        """

        # Compute distance from sample to point
        dist = np.linalg.norm(sample - self.point)
        # Regularization term
        regularization = self.regularization_weight * np.linalg.norm(sample)
        return [dist + regularization, dist, regularization]

    def get_cost_labels(self):
        """Labels for the different cost components.

        CostFunction.evaluate() may return an array of costs. The first one cost[0] is
        always the sum of the other ones, i.e. costs[0] = sum(costs[1:]). This function
        optionally returns labels for the individual cost components.
        """
        return ["dist", "regularization"]


def main():
    """ Main function of the script. """
    directory = Path(tempfile.gettempdir(), "dmpbbo", "demoBboMultipleCostComponents")
    if len(sys.argv) > 1:
        directory = sys.argv[1]

    n_dims = 2
    minimum = np.full(n_dims, 2.0)
    regularization_weight = 1.0
    cost_function = DemoCostFunctionDistanceToPoint(minimum, regularization_weight)

    mean_init = np.full(n_dims, 5.0)
    covar_init = 4.0 * np.eye(n_dims)
    distribution = DistributionGaussian(mean_init, covar_init)

    updater = UpdaterCovarDecay(eliteness=10, weighting_method="PI-BB", covar_decay_factor=0.8)

    n_samples_per_update = 20
    n_updates = 40

    session = run_optimization(
        cost_function, distribution, updater, n_updates, n_samples_per_update, directory
    )

    session.plot()
    plt.show()


if __name__ == "__main__":
    main()
