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


from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import os, sys

# Include scripts for plotting
lib_path = os.path.abspath("../../../python/")
sys.path.append(lib_path)

from functionapproximators.functionapproximators_plotting import *
from functionapproximators.BasisFunction import Gaussian


if __name__ == "__main__":
    """Run some training sessions and plot results."""

    n_centers = 5
    n_samples = 51

    fig = plt.figure(figsize=(12, 12))
    for normalized in [True, False]:
        for n_dims in [1, 2]:

            if n_dims == 1:
                centers = np.linspace(0.0, 2.0, n_centers)
                widths = 0.3 * np.ones(n_centers)
                inputs = np.linspace(-0.5, 2.5, n_samples)
                n_samples_per_dim = n_samples

            else:
                x = np.linspace(0.0, 2.0, n_centers)
                xv, yv = np.meshgrid(x, x)
                centers = np.column_stack((xv.flatten(), yv.flatten()))
                widths = 0.3 * np.ones(centers.shape)

                x = np.linspace(-0.5, 2.5, n_samples)
                xv, yv = np.meshgrid(x, x)
                inputs = np.column_stack((xv.flatten(), yv.flatten()))

                n_samples_per_dim = [n_samples, n_samples]

            kernel_acts = Gaussian.activations(centers, widths, inputs, normalized)

            subplot = 220 + n_dims + (0 if normalized else 2)
            if n_dims == 1:
                ax = fig.add_subplot(subplot)
            else:
                ax = fig.add_subplot(subplot, projection="3d")

            plotBasisFunctions(inputs, kernel_acts, ax, n_samples_per_dim)

    plt.show()
