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
""" Module for the FunctionApproximator class. """

from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from dmpbbo.functionapproximators.Parameterizable import Parameterizable


class FunctionApproximator(Parameterizable):
    """Base class for all function approximators.

    See https://github.com/stulp/dmpbbo/blob/master/tutorial/functionapproximators.md
    """

    def __init__(self, meta_params, model_param_names):
        self._meta_params = meta_params
        self._model_params = {name: None for name in model_param_names}
        self._dim_input = None
        self._selected_param_names = None
        self._inputs_training = None
        self._targets_training = None

    def train(self, inputs, targets, **kwargs):
        """ Train the function approximator with data.

        @param inputs: Input data (n_samples X n_dims_input)
        @param targets: Target data (n_samples X n_dims_output)
        @return: Model parameters of the function approximator.
        """
        # Ensure second dimension, i.e. shape = (30,) => (30,1)
        inputs = inputs.reshape(inputs.shape[0], -1)
        self._model_params = self._train(inputs, targets, self._meta_params, **kwargs)
        self._dim_input = inputs.shape[1]
        if kwargs.get("save_training_data", False):
            self._inputs_training = inputs
            self._targets_training = targets
        return self._model_params

    def predict(self, inputs):
        """ Make predictions for (new) input data.

        @param inputs: Input data (n_samples X n_dims_input)
        @return: Predictions (n_samples X n_dims_output)
        """
        if not self.is_trained():
            raise ValueError("Calling predict() on untrained function approx.")

        # Ensure n_dims=2, i.e. shape = (30,) => (30,1)
        inputs = inputs.reshape(inputs.shape[0], -1)
        if inputs.shape[1] != self._dim_input:
            raise ValueError(
                "Dimensionality of inputs for predict() must be the same as for train()."
            )

        return self._predict(inputs, self._model_params)

    def dim_input(self):
        """ Return the dimensionality of the inputs of the function_approximator
        @return: The dimensionality of the inputs of the function_approximator.
        """
        return self._dim_input

    def is_trained(self):
        """Determine whether the function approximator has already been trained with data or not.
        @return: bool: True if the function approximator has already been trained, False otherwise.
        """
        return isinstance(self._dim_input, int)

    @staticmethod
    @abstractmethod
    def _train(inputs, targets, meta_params, **kwargs):
        """Train the function approximator with input and target examples.

        @param inputs: Input values of the training examples.
        @param targets: Target values of the training examples.
        """
        pass

    @staticmethod
    @abstractmethod
    def _predict(inputs, model_params):
        """Query the function approximator to make a prediction.

        @param inputs: Input values of the query.
        @return: Predicted output values.
        """
        pass

    def set_selected_param_names(self, names):
        """ Set the selected parameters.

        @param names: Name of the parameter to select.
        """
        if isinstance(names, str):
            names = [names]  # Convert to list
        self._selected_param_names = names

    def get_param_vector(self):
        if self._selected_param_names is None:
            return np.array(0)

        """Get a vector containing the values of the selected parameters."""
        if not self.is_trained():
            raise ValueError("FunctionApproximator is not trained.")

        values = []
        for label in self._selected_param_names:
            values.extend(self._model_params[label].flatten())
        return np.asarray(values)

    def set_param_vector(self, values):
        """Set a vector containing the values of the selected parameters."""
        if not self.is_trained():
            raise ValueError("FunctionApproximator is not trained.")

        if len(values) != self.get_param_vector_size():
            raise ValueError(
                f"values ({len(values)}) should have same size as size of selected parameter vector"
                f"({self.get_param_vector_size()}) "
            )

        offset = 0
        for label in self._selected_param_names:
            expected_shape = self._model_params[label].shape
            cur_n_values = np.prod(expected_shape)
            cur_values = values[offset : offset + cur_n_values]
            self._model_params[label] = np.reshape(cur_values, expected_shape)
            offset += cur_n_values

    def get_param_vector_size(self):
        """Get the size of the vector containing the values of the selected parameters."""
        if self._selected_param_names is None:
            return 0
        size = 0
        for label in self._selected_param_names:
            if label in self._model_params:
                size += np.prod(self._model_params[label].shape)
        return size

    def _get_axis(self, fig=None):
        if not fig:
            fig = plt.figure(figsize=(6, 6))
        if self.dim_input() == 1:
            return fig.add_subplot(111)
        elif self.dim_input() == 2:
            return fig.add_subplot(111, projection=Axes3D.name)
        else:
            raise ValueError(f"Cannot create axis with dim_input = {self.dim_input()}")

    @staticmethod
    def _get_grid(inputs_min, inputs_max, n_samples_per_dim=None):
        n_dims = inputs_min.size
        if n_dims == 1:
            if n_samples_per_dim is None:
                n_samples_per_dim = 101
            inputs_grid = np.linspace(inputs_min, inputs_max, n_samples_per_dim)

        elif n_dims == 2:
            if n_samples_per_dim is None:
                n_samples_per_dim = np.atleast_1d([21, 21])
            n_samples = np.prod(n_samples_per_dim)
            # Here comes naive inefficient implementation...
            x1s = np.linspace(inputs_min[0], inputs_max[0], n_samples_per_dim[0])
            x2s = np.linspace(inputs_min[1], inputs_max[1], n_samples_per_dim[1])
            inputs_grid = np.zeros((n_samples, n_dims))
            ii = 0
            for x1 in x1s:
                for x2 in x2s:
                    inputs_grid[ii, 0] = x1
                    inputs_grid[ii, 1] = x2
                    ii += 1
        else:
            raise ValueError(f"Cannot create axis with n_dims = {n_dims}.")

        return inputs_grid, n_samples_per_dim

    @staticmethod
    def _plot_grid_values(inputs, activations, ax, n_samples_per_dim):
        """Plot basis functions activations, whilst being smart about the dimensionality of input
        data. """

        n_dims = len(np.atleast_1d(inputs[0]))
        if n_dims == 1:
            lines = ax.plot(inputs, activations, "-")

        elif n_dims == 2:
            n_basis_functions = len(activations[0])
            basis_functions_to_plot = range(n_basis_functions)

            inputs_0_on_grid = np.reshape(inputs[:, 0], n_samples_per_dim)
            inputs_1_on_grid = np.reshape(inputs[:, 1], n_samples_per_dim)
            lines = []
            # values_range = np.amax(activations) - np.amin(activations)
            for i_basis_function in basis_functions_to_plot:
                # cur_color = colors[i_basis_function]
                cur_activations = activations[:, i_basis_function]
                # Make plotting easier by leaving out small numbers
                # cur_activations[numpy.abs(cur_activations)<values_range*0.001] = numpy.nan
                activations_on_grid = np.reshape(cur_activations, n_samples_per_dim)
                cur_lines = ax.plot_wireframe(
                    inputs_0_on_grid,
                    inputs_1_on_grid,
                    activations_on_grid,
                    linewidth=0.5,
                    rstride=1,
                    cstride=1,
                )
                lines.append(cur_lines)

        else:
            print(f"Cannot plot input data with a dimensionality of {n_dims}")
            lines = []

        return lines

    @abstractmethod
    def plot_model_parameters(self, inputs_min, inputs_max, **kwargs):
        """ Plot a representation of the model parameters on a grid.

        @param inputs_min: The min values for the grid
        @param inputs_max:  The max values for the grid
        @return: line handles and axis
        """
        pass

    def plot_predictions_grid(self, inputs_min, inputs_max, **kwargs):
        """ Plot the predictions of the function approximator on a grid of input values.

        @param inputs_min: The min values for the grid
        @param inputs_max:  The max values for the grid
        @return: line handles and axis
        """
        ax = kwargs.get("ax") or self._get_axis()

        inputs, n_samples_per_dim = FunctionApproximator._get_grid(inputs_min, inputs_max)
        outputs = self.predict(inputs)

        h = []
        if self.dim_input() == 1:
            h = ax.plot(inputs, outputs, "-")
        elif self.dim_input() == 2:
            inputs_0_on_grid = np.reshape(inputs[:, 0], n_samples_per_dim)
            inputs_1_on_grid = np.reshape(inputs[:, 1], n_samples_per_dim)
            outputs_on_grid = np.reshape(outputs, n_samples_per_dim)
            h = ax.plot_wireframe(
                inputs_0_on_grid, inputs_1_on_grid, outputs_on_grid, rstride=1, cstride=1
            )
        else:
            print("Cannot plot input data with a dimensionality of {self.dim_input()}")

        plt.setp(h, linewidth=1, color=[0.3, 0.3, 0.9], alpha=0.5)

        return h, ax

    def plot_predictions(self, inputs, **kwargs):
        """ Plot the predictions of a function approximator for given inputs.

        @param inputs: The input samples (n_samples X n_input_dims )
        @return: line handles and axis
        """
        targets = kwargs.get("targets", [])
        ax = kwargs.get("ax") or self._get_axis()
        plot_residuals = kwargs.get("plot_residuals", True)

        outputs = self.predict(inputs)

        # If only few data points, plot individual markers, otherwise plot lines
        line_style = "o" if inputs.shape[0] < 40 else "-"

        h_residuals = []
        h_targets = []
        if self.dim_input() == 1:
            if len(targets) > 0:
                h_targets = ax.plot(inputs, targets, line_style)
                if plot_residuals:
                    for ii in range(len(inputs)):
                        x = [inputs[ii], inputs[ii]]
                        y = [targets[ii], outputs[ii]]
                        h = ax.plot(x, y, "-")
                        h_residuals.append(h)
            h_outputs = ax.plot(inputs, outputs, line_style)

        elif self.dim_input() == 2:
            if len(targets) > 0:
                h_targets = ax.plot(inputs[:, 0], inputs[:, 1], targets, line_style)
                if plot_residuals:
                    for ii in range(len(inputs)):
                        x0 = [inputs[ii, 0], inputs[ii, 0]]
                        x1 = [inputs[ii, 1], inputs[ii, 1]]
                        y = [targets[ii], outputs[ii]]
                        h = ax.plot(x0, x1, y, "-")
                        h_residuals.append(h)
            h_outputs = ax.plot(inputs[:, 0], inputs[:, 1], outputs, line_style)

        else:
            raise ValueError(f"Cannot plot input data with dim_input() = {self.dim_input()}")

        if len(targets) > 0:
            plt.setp(h_targets, markeredgecolor=None, markerfacecolor=[0.7, 0.7, 0.7], markersize=8)
        plt.setp(h_outputs, markeredgecolor=None, markerfacecolor=[0.2, 0.2, 0.8], markersize=6)
        if len(targets) > 0:
            plt.setp(h_residuals, color=[0.8, 0.3, 0.3], linewidth=2)

        return h_outputs, ax

    def plot(self, inputs=None, **kwargs):
        """ Plot the predictions of a function approximator for given inputs.

        @param inputs: The input samples (n_samples X n_input_dims )
        @return: line handles and axis
        """
        if inputs is None:
            if self._inputs_training is None:
                print("WARNING: no inputs provided, and inputs for training are not available. "
                      "Not plotting")
                return None, None
            else:
                # Plot for the inputs that were used for training
                inputs = self._inputs_training

        ax = kwargs.get("ax") or self._get_axis()
        targets = kwargs.get("targets", self._targets_training)
        plot_residuals = kwargs.get("plot_residuals", True)
        plot_model_parameters = kwargs.get("plot_model_parameters", False)

        inputs_min = np.min(inputs, axis=0)
        inputs_max = np.max(inputs, axis=0)
        self.plot_predictions(inputs, targets=targets, ax=ax, plot_residuals=plot_residuals)
        if plot_model_parameters:
            self.plot_model_parameters(inputs_min, inputs_max, ax=ax)
        return self.plot_predictions_grid(inputs_min, inputs_max, ax=ax)
