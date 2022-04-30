# This file is part of DmpBbo, a set of libraries and programs for the
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2018 Freek Stulp
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


class FunctionApproximatorLWR(FunctionApproximator):
    def __init__(self, n_bfs_per_dim, intersection_height=0.5, regularization=0.0):

        meta_params = {
            "n_basis_functions_per_dim": np.atleast_1d(n_bfs_per_dim),
            "intersection_height": intersection_height,
            "regularization": regularization,
        }

        model_param_names = ["centers", "widths", "slopes", "offsets"]

        super().__init__(meta_params, model_param_names)

    @staticmethod
    def _train(inputs, targets, meta_params, **kwargs):

        # Determine the centers and widths of the basis functions, given the input data range
        n_bfs_per_dim = meta_params["n_basis_functions_per_dim"]
        height = meta_params["intersection_height"]
        centers, widths = Gaussian.get_centers_and_widths(inputs, n_bfs_per_dim, height)
        model_params = {"centers": centers, "widths": widths}

        # Prepare weighted least squares regressions
        n_bfs = np.prod(n_bfs_per_dim)
        n_dims = centers.shape[1]
        slopes = np.ones([n_bfs, n_dims])
        offsets = np.ones([n_bfs, 1])

        # Prepare the weighted-least squares function approximator
        reg = meta_params["regularization"]
        wls_meta_params = {"regularization": reg, "use_offset": True}
        fa_lws = FunctionApproximatorWLS(wls_meta_params)

        # Perform one weighted least squares regression for each kernel
        activations = FunctionApproximatorLWR._activations(inputs, model_params)
        for i_kernel in range(n_bfs):
            # Do one weighted least-squares regression with the current weights
            weights = activations[:, i_kernel]
            wls_model_params = fa_lws.train(inputs, targets, weights=weights)

            slopes[i_kernel, :] = wls_model_params["slope"]
            offsets[i_kernel] = wls_model_params["offset"]

        model_params["offsets"] = offsets
        model_params["slopes"] = slopes

        return model_params

    @staticmethod
    def _activations(inputs, model_params):
        """Get the activations of the basis functions.

        Uses the centers and widths in the model parameters.

        Args:
            inputs (numpy.ndarray): Input values of the query.
        """
        return Gaussian.activations(
            inputs, centers=model_params["centers"], widths=model_params["widths"], normalized=True
        )

    @staticmethod
    def _get_lines(inputs, model_params):
        # Ensure n_dims=2, i.e. shape = (30,) => (30,1)
        inputs = inputs.reshape(inputs.shape[0], -1)

        slopes = model_params["slopes"]
        offsets = model_params["offsets"]

        # Compute the line segments
        n_lines = offsets.size
        n_samples = inputs.shape[0]
        lines = np.zeros([n_samples, n_lines])
        for i_line in range(n_lines):
            # Apparently, not everybody has python3.5 installed, so don't use @
            # lines[:,i_line] =  inputs@slopes[i_line,:].T + offsets[i_line]
            lines[:, i_line] = np.dot(inputs, slopes[i_line, :].T) + offsets[i_line]

        return lines

    @staticmethod
    def _predict(inputs, model_params):
        # Ensure n_dims=2, i.e. shape = (30,) => (30,1)
        inputs = inputs.reshape(inputs.shape[0], -1)

        lines = FunctionApproximatorLWR._get_lines(inputs, model_params)

        # Weight the values for each line with the normalized basis function activations
        # Get the activations of the basis functions
        activations = FunctionApproximatorLWR._activations(inputs, model_params)

        outputs = (lines * activations).sum(axis=1)
        return outputs

    def plot_model_parameters(self, inputs_min, inputs_max, **kwargs):

        inputs, n_samples_per_dim = FunctionApproximator._get_grid(inputs_min, inputs_max)
        activations = self._activations(inputs, self._model_params)

        ax = kwargs.get("ax") or self._get_axis()

        lines = self._plot_grid_values(inputs, activations, ax, n_samples_per_dim)
        alpha = 1.0 if self.dim_input() < 2 else 0.3
        plt.setp(lines, color=[0.7, 0.7, 0.7], linewidth=1, alpha=alpha)

        # Plot lines also
        line_values = self._get_lines(inputs, self._model_params)

        # Plot line segment only when a basis function is the most active
        # values_range = np.amax(activations) - np.amin(activations)
        n_basis_functions = activations.shape[1]
        max_activations = np.max(activations, axis=1)
        for i_bf in range(n_basis_functions):
            cur_activations = activations[:, i_bf]
            smaller = cur_activations < 0.2 * max_activations
            line_values[smaller, i_bf] = np.nan

        lines = self._plot_grid_values(inputs, line_values, ax, n_samples_per_dim)
        alpha = 1.0 if self.dim_input() < 2 else 0.5
        w = 4 if self.dim_input() < 2 else 1
        plt.setp(lines, color=[0.8, 0.8, 0.8], linewidth=w, alpha=alpha)

        return lines, ax
