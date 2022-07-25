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
"""Script for bbo demo."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dmpbbo.bbo.CostFunction import CostFunction
from dmpbbo.bbo.DistributionGaussian import DistributionGaussian
from dmpbbo.bbo.run_optimization import run_optimization
from dmpbbo.bbo.updaters import UpdaterCovarAdaptation, UpdaterCovarDecay, UpdaterMean


class DemoCostFunctionDistanceToPoint(CostFunction):
    """ CostFunction in which the distance to a pre-defined point must be minimized."""

    def __init__(self, point):
        """ Constructor.

        @param point: Point to which distance must be minimized.
        """
        self.point = np.asarray(point)

    def evaluate(self, sample):
        """The cost function which defines the cost_function.

        Computes the distance from sample to point

         @param sample: The sample
         @return: The scalar cost components for the sample. The first item costs[0] should contain
         the total cost.
        """
        return [np.linalg.norm(sample - self.point)]


def main():
    """ Main function of the script. """
    directory = sys.argv[1] if len(sys.argv) > 1 else None

    n_dims = 2
    minimum = np.full(n_dims, 2.0)
    cost_function = DemoCostFunctionDistanceToPoint(minimum)

    updaters = {}  # noqa

    updaters["fixed_exploration"] = UpdaterMean(eliteness=10, weighting_method="PI-BB")

    updaters["covar_decay"] = UpdaterCovarDecay(
        eliteness=10, weighting_method="PI-BB", covar_decay_factor=0.8
    )

    updaters["covar_adaptation"] = UpdaterCovarAdaptation(
        eliteness=10,
        weighting_method="PI-BB",
        max_level=None,
        min_level=0.000001,
        diag_only=False,
        learning_rate=0.8,
    )

    for name, updater in updaters.items():
        print("Distribution updater: " + name)

        mean_init = np.full(n_dims, 5.0)
        covar_init = 1.0 * np.eye(n_dims)
        distribution = DistributionGaussian(mean_init, covar_init)

        n_samples_per_update = 20
        n_updates = 40

        cur_directory = None
        if directory:
            cur_directory = Path(directory, name)

        session = run_optimization(
            cost_function, distribution, updater, n_updates, n_samples_per_update, cur_directory
        )
        fig = session.plot()
        fig.canvas.set_window_title(f"Optimization with covar_update={name}")

    plt.show()


if __name__ == "__main__":
    main()
