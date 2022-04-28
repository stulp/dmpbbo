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


import sys

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
        
        Args:
            point: Point to which distance must be minimized.
        """
        self.point = np.asarray(point)
        self.regularization_weight = regularization_weight

    def evaluate(self, sample):
        # Compute distance from sample to point
        dist = np.linalg.norm(sample - self.point)
        # Regularization term
        regularization = self.regularization_weight * np.linalg.norm(sample)
        return [dist + regularization, dist, regularization]

    def cost_labels(self):
        return ["dist", "regularization"]


if __name__ == "__main__":

    directory = "/tmp/dmpbbo/demoBboMultipleCostComponents"
    if len(sys.argv) > 1:
        directory = sys.argv[1]

    n_dims = 2
    minimum = np.full(n_dims, 2.0)
    regularization_weight = 1.0
    cost_function = DemoCostFunctionDistanceToPoint(minimum, regularization_weight)

    mean_init = np.full(n_dims, 5.0)
    covar_init = 4.0 * np.eye(n_dims)
    distribution = DistributionGaussian(mean_init, covar_init)

    eliteness = 10
    weighting_method = "PI-BB"
    covar_decay_factor = 0.8
    updater = UpdaterCovarDecay(eliteness, weighting_method, covar_decay_factor)

    n_samples_per_update = 20
    n_updates = 40

    session = run_optimization(
        cost_function, distribution, updater, n_updates, n_samples_per_update, directory
    )

    session.plot()
    plt.show()
