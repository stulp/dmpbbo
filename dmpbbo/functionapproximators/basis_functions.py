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

from abc import ABC, abstractmethod

import numpy as np


class BasisFunction(ABC):
    @staticmethod
    @abstractmethod
    def activations(inputs, **kwargs):
        pass


class Gaussian(BasisFunction):
    @staticmethod
    def activations(inputs, **kwargs):
        return Gaussian._activations(
            inputs, kwargs.get("centers"), kwargs.get("widths"), kwargs.get("normalized")
        )

    @staticmethod
    def _activations(inputs, centers, widths, normalized_basis_functions=False):
        """Get the activations for given centers, widths and inputs.

        Args:
            centers: The center of the basis function (size: n_basis_functions X n_dims)
            widths: The width of the basis function (size: n_basis_functions X n_dims)
            inputs: The input data (size: n_samples X n_dims)
            normalized_basis_functions: Whether to normalize the basis functions (default=False)

        Returns: The kernel activations, computed for each of the samples in the input data (size: n_samples X
        n_basis_functions)
        """

        n_samples = inputs.shape[0]
        n_basis_functions = centers.shape[0]
        n_dims = centers.shape[1] if len(centers.shape) > 1 else 1

        if n_dims == 1:
            # Make sure arguments have shape (N,1) not (N,)
            centers = centers.reshape(n_basis_functions, 1)
            widths = widths.reshape(n_basis_functions, 1)
            inputs = inputs.reshape(n_samples, 1)

        kernel_activations = np.ones([n_samples, n_basis_functions])

        if normalized_basis_functions and n_basis_functions == 1:
            # Normalizing one Gaussian basis function with itself leads to 1 everywhere.
            kernel_activations.fill(1.0)
            return kernel_activations

        for bb in range(n_basis_functions):
            # Here, we compute the values of a (unnormalized) multi-variate Gaussian:
            #   activation = exp(-0.5*(x-mu)*Sigma^-1*(x-mu))
            # Because Sigma is diagonal in our case, this simplifies to
            #   activation = exp(\sum_d=1^D [-0.5*(x_d-mu_d)^2/Sigma_(d,d)])
            #              = \prod_d=1^D exp(-0.5*(x_d-mu_d)^2/Sigma_(d,d))
            # This last product is what we compute below incrementally

            for i_dim in range(n_dims):
                c = centers[bb, i_dim]
                w = widths[bb, i_dim]
                for i_s in range(n_samples):
                    x = inputs[i_s, i_dim]
                    kernel_activations[i_s, bb] *= np.exp(-0.5 * np.square(x - c) / (w * w))

        if normalized_basis_functions:
            # Normalize the basis value; they should sum to 1.0 for each time step.
            for i_sample in range(n_samples):
                sum_kernel_activations = kernel_activations[i_sample, :].sum()
                for i_basis in range(n_basis_functions):
                    if sum_kernel_activations == 0.0:
                        # Apparently, no basis function was active. Set all to same value
                        kernel_activations[i_sample, i_basis] = 1.0 / n_basis_functions
                    else:
                        # Standard case, normalize so that they sum to 1.0
                        kernel_activations[i_sample, i_basis] /= sum_kernel_activations

        return kernel_activations

    @staticmethod
    def get_centers_and_widths(inputs, n_bfs_per_dim, intersection_height=0.7):
        """Get the centers and widths of basis functions.

        Args:
            inputs: The input data (size: n_samples X n_dims)
            n_bfs_per_dim: Number of basis functions per input dimension.
            intersection_height: The relative value at which two neighbouring basis functions will intersect (
            default=0.7)

        Returns:
            centers: Centers of the basis functions (matrix of size n_basis_functions X n_input_dims
            widths: Widths of the basis functions (matrix of size n_basis_functions X n_input_dims
        """
        min_vals = inputs.min(axis=0)
        max_vals = inputs.max(axis=0)
        min_vals = np.atleast_1d(min_vals)
        max_vals = np.atleast_1d(max_vals)
        n_dims = len(min_vals)
        n_bfs_per_dim = np.atleast_1d(n_bfs_per_dim)
        if n_bfs_per_dim.size < n_dims:
            if n_bfs_per_dim.size == 1:
                n_bfs_per_dim = n_bfs_per_dim * np.ones(n_dims).astype(int)
            else:
                raise ValueError(f"n_bfs_per_dim should be of size {n_dims}")

        centers_per_dim_local = []
        widths_per_dim_local = []
        for i_dim in range(n_dims):
            n_bfs = n_bfs_per_dim[i_dim]

            cur_centers = np.linspace(min_vals[i_dim], max_vals[i_dim], n_bfs)

            # Determine the widths from the centers
            cur_widths = np.ones(n_bfs)
            h = intersection_height
            if n_bfs > 1:
                # Consider two neighbouring basis functions, exp(-0.5(x-c0)^2/w^2) and exp(-0.5(x-c1)^2/w^2)
                # Assuming the widths are the same for both, they are certain to intersect at x = 0.5(c0+c1)
                # And we want the activation at x to be 'intersection'. So
                #            y = exp(-0.5(x-c0)^2/w^2)
                # intersection = exp(-0.5((0.5(c0+c1))-c0)^2/w^2)
                # intersection = exp(-0.5((0.5*c1-0.5*c0)^2/w^2))
                # intersection = exp(-0.5((0.5*(c1-c0))^2/w^2))
                # intersection = exp(-0.5(0.25*(c1-c0)^2/w^2))
                # intersection = exp(-0.125((c1-c0)^2/w^2))
                #            w = sqrt((c1-c0)^2/-8*ln(intersection))
                for cc in range(n_bfs - 1):
                    w = np.sqrt(np.square(cur_centers[cc + 1] - cur_centers[cc]) / (-8 * np.log(h)))
                    cur_widths[cc] = w

                cur_widths[n_bfs - 1] = cur_widths[n_bfs - 2]

            centers_per_dim_local.append(cur_centers)
            widths_per_dim_local.append(cur_widths)

        # We now have the centers and widths for each dimension separately.
        # This is like meshgrid.flatten, but then for any number of dimensions
        # I'm sure numpy has better functions for this, but I could not find them, and I already
        # had the code in C++.
        digit_max = n_bfs_per_dim
        n_centers = np.prod(digit_max)
        digit = [0] * n_dims

        centers = np.zeros((n_centers, n_dims))
        widths = np.zeros((n_centers, n_dims))
        i_center = 0
        while digit[0] < digit_max[0]:
            for i_dim in range(n_dims):
                centers[i_center, i_dim] = centers_per_dim_local[i_dim][digit[i_dim]]
                widths[i_center, i_dim] = widths_per_dim_local[i_dim][digit[i_dim]]
            i_center += 1

            # Increment last digit by one
            digit[n_dims - 1] += 1
            for i_dim in range(n_dims - 1, 0, -1):
                if digit[i_dim] >= digit_max[i_dim]:
                    digit[i_dim] = 0
                    digit[i_dim - 1] += 1

        return centers, widths


class Cosine(BasisFunction):
    @staticmethod
    def activations(inputs, **kwargs):
        raise NotImplementedError("Sorry: Cosine BasisFunction not implemented yet.")
