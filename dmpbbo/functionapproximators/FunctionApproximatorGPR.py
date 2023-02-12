# This file is part of DmpBbo, a set of libraries and programs for the
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2022 Freek Stulp
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


class FunctionApproximatorGPR(FunctionApproximator):
    """Function approximator based on Gaussian Process Regression (GPR).
    """

    def __init__(self, max_covariance=1.0, lengths=1.0):
        """Constructor for GPR function approximator

        Args:
            max_covariance (double): The maximum allowable covariance of the covar function (aka sigma in Ebden'15)
            lengths (scalar or numpy.array:  Standard deviation in the isotropic covariance function, i.e.
            e^(-0.5*(x-x}')^T * W * (x-x')), with W = lengths^2 * I
        """
        meta_params = {"max_covariance": max_covariance, "lengths": np.atleast_1d(lengths)}
        model_param_names = ["gram_inv", "gram_inv_targets"]
        super().__init__(meta_params, model_param_names)

    @staticmethod
    def _activations(inputs, model_params):
        activations = Gaussian.activations(
            inputs, centers=model_params["inputs"], widths=model_params["lengths"], normalized=False
        )
        return model_params["max_covariance"] * activations

    @staticmethod
    def _train(inputs, targets, meta_params, **kwargs):

        model_params = {
            "inputs": inputs,
            "lengths": np.full(inputs.shape, meta_params["lengths"]),
            "max_covariance": meta_params["max_covariance"],
        }

        # Compute the Gram matrix (every input point is itself a center)
        gram = FunctionApproximatorGPR._activations(inputs, model_params)

        gram_inv = np.linalg.inv(gram)
        gram_inv_targets = gram_inv @ targets

        model_params["gram_inv_targets"] = gram_inv_targets  # Required to compute mean
        # model_params["gram_inv"] = gram_inv  # Required to compute variance

        return model_params

    @staticmethod
    def _predict(inputs, model_params):

        weights = model_params["gram_inv_targets"]
        activations = FunctionApproximatorGPR._activations(inputs, model_params)

        weighted_acts = np.zeros(activations.shape)
        for ii in range(activations.shape[1]):
            weighted_acts[:, ii] = activations[:, ii] * weights[ii]

        return weighted_acts.sum(axis=1)

    def plot_model_parameters(self, inputs_min, inputs_max, **kwargs):
        """ Plot a representation of the model parameters on a grid.

        @param inputs_min: The min values for the grid
        @param inputs_max:  The max values for the grid
        @return: line handles and axis
        """
        inputs, n_samples_per_dim = FunctionApproximator._get_grid(inputs_min, inputs_max)

        weights = self._model_params["gram_inv_targets"]
        activations = FunctionApproximatorGPR._activations(inputs, self._model_params)

        weighted_acts = np.zeros(activations.shape)
        for ii in range(activations.shape[1]):
            weighted_acts[:, ii] = activations[:, ii] * weights[ii]

        ax = kwargs.get("ax") or self._get_axis()

        # lines = self._plot_grid_values(inputs, activations, ax, n_samples_per_dim)
        lines = self._plot_grid_values(inputs, weighted_acts, ax, n_samples_per_dim)
        alpha = 1.0 if self.dim_input() < 2 else 0.3
        plt.setp(lines, linestyle="--", color=[0.7, 0.7, 0.7], linewidth=2, alpha=alpha)

        return lines, ax
