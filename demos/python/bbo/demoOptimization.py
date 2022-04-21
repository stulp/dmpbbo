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


import os
import sys
import numpy as np

# Add relative path, in case PYTHONPATH is not set
lib_path = os.path.abspath("../../python/")
sys.path.append(lib_path)

from bbo.CostFunction import CostFunction
from bbo.DistributionGaussian import DistributionGaussian
from bbo.updaters import *
from bbo.runOptimization import runOptimization


class DemoCostFunctionDistanceToPoint(CostFunction):
    """ CostFunction in which the distance to a pre-defined point must be minimized."""

    def __init__(self, point):
        """ Constructor.
        \param[in] point Point to which distance must be minimized.
        """
        self.point = np.asarray(point)

    def evaluate(self, sample):
        # Compute distance from sample to point
        return [np.linalg.norm(sample - self.point)]


if __name__ == "__main__":

    directory = None
    if len(sys.argv) > 1:
        directory = sys.argv[1]

    n_dims = 2
    minimum = np.full(n_dims, 2.0)
    cost_function = DemoCostFunctionDistanceToPoint(minimum)

    fig_counter = 0
    for covar_update in ["none", "decay", "adaptation"]:
        print(covar_update)

        mean_init = np.full(n_dims, 5.0)
        covar_init = 4.0 * np.eye(n_dims)
        distribution = DistributionGaussian(mean_init, covar_init)

        eliteness = 10
        weighting_method = "PI-BB"  # or 'CEM' or 'CMA-ES'
        if covar_update == "none":
            updater = UpdaterMean(eliteness, weighting_method)
        elif covar_update == "decay":
            covar_decay_factor = 0.8
            updater = UpdaterCovarDecay(eliteness, weighting_method, covar_decay_factor)
        else:
            min_level = 0.000001
            max_level = None
            diag_only = False
            learning_rate = 0.75
            updater = UpdaterCovarAdaptation(
                eliteness,
                weighting_method,
                max_level,
                min_level,
                diag_only,
                learning_rate,
            )

        n_samples_per_update = 20
        n_updates = 40

        import matplotlib.pyplot as plt

        fig = plt.figure(fig_counter, figsize=(15, 5))
        fig.canvas.set_window_title("Optimization with covar_update=" + covar_update)
        fig_counter += 1

        cur_directory = directory
        if cur_directory != None:
            cur_directory += "/" + covar_update

        learning_curve = runOptimization(
            cost_function,
            distribution,
            updater,
            n_updates,
            n_samples_per_update,
            fig,
            cur_directory,
        )

    plt.show()
