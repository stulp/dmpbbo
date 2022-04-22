# This file is part of DmpBbo, a set of libraries and programs for the
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2018, 2022 Freek Stulp
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

import os, sys

lib_path = os.path.abspath("../../../python/")
sys.path.append(lib_path)

from functionapproximators.FunctionApproximator import FunctionApproximator
from functionapproximators.BasisFunction import *
from functionapproximators.leastSquares import *


class FunctionApproximatorRBFN(FunctionApproximator):
    def __init__(self, n_bfs_per_dim, intersection_height=0.7, regularization=0.0):
        """Initialize an RBNF function approximator.

        Args:
            n_bfs_per_dim Number of basis functions per input dimension.
            intersection_height The relative value at which two neighbouring basis functions will intersect (default=0.7)
            regularization Regularization parameter (default=0.0)
        """
        meta_params = {
            "n_basis_functions_per_dim": np.atleast_1d(n_bfs_per_dim),
            "intersection_height": intersection_height,
            "regularization": regularization,
        }

        model_param_names = ["centers", "widths", "weights"]

        super().__init__(meta_params, model_param_names)

    @staticmethod
    def _train(inputs, targets, meta_params):

        # Determine the centers and widths of the basis functions, given the input data range
        n_bfs_per_dim = meta_params["n_basis_functions_per_dim"]
        height = meta_params["intersection_height"]
        centers, widths = Gaussian.getCentersAndWidths(inputs, n_bfs_per_dim, height)
        model_params = {"centers": centers, "widths": widths}

        # Get the activations of the basis functions
        activations = FunctionApproximatorRBFN._getActivations(inputs, model_params)

        # Perform one least squares regression
        use_offset = False
        reg = meta_params["regularization"]
        weights = leastSquares(activations, targets, use_offset, reg)

        n_bfs = centers.shape[0]
        model_params["weights"] = weights.reshape(n_bfs, -1)

        return model_params

    @staticmethod
    def _getActivations(inputs, model_params):
        normalize = False
        return Gaussian.activations(
            model_params["centers"], model_params["widths"], inputs, normalize
        )

    @staticmethod
    def _predict(inputs, model_params):
        acts = FunctionApproximatorRBFN._getActivations(inputs, model_params)
        weighted_acts = np.zeros(acts.shape)
        for ii in range(acts.shape[1]):
            weighted_acts[:, ii] = acts[:, ii] * model_params["weights"][ii]
        return weighted_acts.sum(axis=1)
