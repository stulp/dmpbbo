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
import numpy as np
from matplotlib import pyplot as plt

from dmpbbo.functionapproximators.basis_functions import Gaussian
from dmpbbo.functionapproximators.FunctionApproximator import FunctionApproximator
from dmpbbo.functionapproximators.FunctionApproximatorWLS import FunctionApproximatorWLS


class FunctionApproximatorRBFN(FunctionApproximator):
    def __init__(self, n_bfs_per_dim, intersection_height=0.7, regularization=0.0):
        """Initialize an RBNF function approximator.

        Args:
            n_bfs_per_dim Number of basis functions per input dimension.
            intersection_height Relative value at which two neighbouring basis functions
            will intersect (default=0.7)
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
    def _train(inputs, targets, meta_params, **kwargs):
        # Determine the centers and widths of the basis functions, given the input data range
        n_bfs_per_dim = meta_params["n_basis_functions_per_dim"]
        n_bfs = np.prod(n_bfs_per_dim)

        height = meta_params["intersection_height"]
        centers, widths = Gaussian.get_centers_and_widths(inputs, n_bfs_per_dim, height)
        model_params = {"centers": centers, "widths": widths}

        # Get the activations of the basis functions
        activations = FunctionApproximatorRBFN._activations(inputs, model_params)

        # Prepare the least squares function approximator and train it
        use_offset = False
        fa_lws = FunctionApproximatorWLS(use_offset, meta_params["regularization"])
        wls_model_params = fa_lws.train(activations, targets)

        model_params["weights"] = wls_model_params["slope"].reshape(n_bfs, -1)

        return model_params

    @staticmethod
    def _activations(inputs, model_params):
        return Gaussian.activations(
            inputs, centers=model_params["centers"], widths=model_params["widths"], normalized=False
        )

    @staticmethod
    def _predict(inputs, model_params):
        acts = FunctionApproximatorRBFN._activations(inputs, model_params)
        weighted_acts = np.zeros(acts.shape)
        for ii in range(acts.shape[1]):
            weighted_acts[:, ii] = acts[:, ii] * model_params["weights"][ii]
        return weighted_acts.sum(axis=1)

    def plot_model_parameters(self, inputs_min, inputs_max, **kwargs):
        inputs, n_samples_per_dim = FunctionApproximator._get_grid(inputs_min, inputs_max)
        activations = self._activations(inputs, self._model_params)

        ax = kwargs.get("ax") or self._get_axis()

        lines = self._plot_grid_values(inputs, activations, ax, n_samples_per_dim)
        alpha = 1.0 if self.dim_input() < 2 else 0.3
        plt.setp(lines, color=[0.7, 0.7, 0.7], linewidth=1, alpha=alpha)

        return lines, ax
