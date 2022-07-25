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
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
""" Script for function_approximators_parameterizable demo."""

import numpy as np
from matplotlib import pyplot as plt

from dmpbbo.functionapproximators.FunctionApproximatorLWR import FunctionApproximatorLWR
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN


def target_function(n_samples_per_dim):
    """ Target-function for training

    @param n_samples_per_dim: Number of samples for each dimension
    @return: inputs and corresponding targets
    """

    n_dims = 1 if np.isscalar(n_samples_per_dim) else len(n_samples_per_dim)

    if n_dims == 1:
        inputs = np.linspace(0.0, 2.0, n_samples_per_dim)
        targets = 3 * np.exp(-inputs) * np.sin(2 * np.square(inputs))

    else:
        n_samples = np.prod(n_samples_per_dim)
        # Here comes naive inefficient implementation...
        x1s = np.linspace(-2.0, 2.0, n_samples_per_dim[0])
        x2s = np.linspace(-2.0, 2.0, n_samples_per_dim[1])
        inputs = np.zeros((n_samples, n_dims))
        targets = np.zeros(n_samples)
        ii = 0
        for x1 in x1s:
            for x2 in x2s:
                inputs[ii, 0] = x1
                inputs[ii, 1] = x2
                targets[ii] = 2.5 * x1 * np.exp(-np.square(x1) - np.square(x2))
                ii += 1

    return inputs, targets


def train(fa_name, n_dims):
    """ Main function of the script. """
    # Generate training data
    n_samples_per_dim = 30 if n_dims == 1 else [10, 10]
    (inputs, targets) = target_function(n_samples_per_dim)

    n_rfs = 9 if n_dims == 1 else [5, 5]  # Number of basis functions. To be used later.

    # Initialize function approximator
    if fa_name == "Locally Weighted Regression":
        # This value for intersection is quite low. But for the demo it is nice
        # because it makes the linear segments quite obvious.
        intersection = 0.2
        fa = FunctionApproximatorLWR(n_rfs, intersection)
    else:
        intersection = 0.7
        fa = FunctionApproximatorRBFN(n_rfs, intersection)

    # Train function approximator with data
    fa.train(inputs, targets)

    # Make predictions for the targets
    outputs = fa.predict(inputs)  # noqa

    if fa_name == "Locally Weighted Regression":
        fa.set_selected_param_names(["offsets", "widths"])
    else:
        fa.set_selected_param_names(["weights", "widths"])
    values = fa.get_param_vector()

    # Plotting
    inputs_min = np.min(inputs, axis=0)
    inputs_max = np.max(inputs, axis=0)
    w = 4 if n_dims == 1 else 2
    a = 1 if n_dims == 1 else 0.5

    fig = plt.figure(figsize=(10, 5))
    if n_dims == 1:
        axs = [fig.add_subplot(121 + i) for i in range(2)]
    else:
        axs = [fig.add_subplot(121 + i, projection="3d") for i in range(2)]

    for noise in ["additive", "multiplicative"]:
        ax = axs[0] if noise == "additive" else axs[1]

        # Original function
        fa.set_param_vector(values)
        h, _ = fa.plot_predictions_grid(inputs_min, inputs_max, ax=ax)
        plt.setp(h, color=[0.0, 0.0, 0.6], linewidth=w, alpha=a)
        if n_dims == 1:
            hb, _ = fa.plot_model_parameters(inputs_min, inputs_max, ax=ax)
            plt.setp(hb, color=[0.6, 0.0, 0.0], linewidth=w, alpha=a)

        # Perturbed function
        for i_sample in range(5):

            if noise == "additive":
                rand_vector = 0.05 * np.random.standard_normal(values.shape)
                new_values = rand_vector + values
            else:
                rand_vector = 1.0 + 0.1 * np.random.standard_normal(values.shape)
                new_values = rand_vector * values
            fa.set_param_vector(new_values)

            if n_dims == 1:
                hb, _ = fa.plot_model_parameters(inputs_min, inputs_max, ax=ax)
                plt.setp(hb, color=[1.0, 0.5, 0.5], linewidth=w / 3)
            h, _ = fa.plot_predictions_grid(inputs_min, inputs_max, ax=ax)
            plt.setp(h, color=[0.5, 0.5, 1.0], linewidth=w / 2)

        ax.set_title(f"{fa_name} {n_dims}D\n({noise} noise)")
        plt.gcf().canvas.set_window_title(f"{fa_name} {n_dims}D")


def main():
    """Run some training sessions and plot results."""

    names = ["Radial Basis Function Network", "Locally Weighted Regression"]
    for fa_name in names:
        for n_dims in [1, 2]:
            train(fa_name, n_dims)

    plt.show()


if __name__ == "__main__":
    main()
