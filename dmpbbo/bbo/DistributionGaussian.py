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


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def plot_error_ellipse(mu, cov, ax=None, **kwargs):
    """
    Plot the error ellipse at a point given its covariance matrix

    Parameters
    ----------
    mu : array (2,)
    The center of the ellipse

    cov : array (2,2)
    The covariance matrix for the point

    ax : matplotlib.Axes, optional
    The axis to plot on

    **kwargs : dict
    These keywords are passed to matplotlib.patches.Ellipse

    From https://github.com/dfm/dfmplot/blob/master/dfmplot/ellipse.py
    """

    facecolor = kwargs.pop("facecolor", "none")
    edgecolor = kwargs.pop("edgecolor", "k")

    x, y = mu
    U, S, V = np.linalg.svd(cov)  # noqa
    theta = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    ellipse_plot = Ellipse(
        xy=(x, y),
        width=2 * np.sqrt(S[0]),
        height=2 * np.sqrt(S[1]),
        angle=float(theta),
        facecolor=facecolor,
        edgecolor=edgecolor,
        **kwargs,
    )

    if not ax:
        ax = plt.gca()
    lines = ax.add_patch(ellipse_plot)
    return lines, ax


class DistributionGaussian:
    """ A class for representing a Gaussian distribution.
    """

    def __init__(self, mean=0.0, covar=1.0):
        """ Construct the Gaussian distribution with a mean and covariance matrix.
        
        Args:
            mean: Mean of the distribution
            covar: Covariance matrix of the distribution
        """
        self.mean = mean
        self.covar = covar

    def generateSamples(self, n_samples=1):
        """ Generate samples from the distribution.
        
        Args:
            n_samples Number of samples to sample
        Returns:
            The samples themselves (size n_samples X dim(mean)
        """
        return np.random.multivariate_normal(self.mean, self.covar, n_samples)

    def maxEigenValue(self):
        """ Get the largest eigenvalue of the covariance matrix of this distribution.
        Returns:
            largest eigenvalue of the covariance matrix of this distribution.
        """
        return max(np.linalg.eigvals(self.covar))

    def __str__(self):
        """ Get a string representation of an object of this class.
        Returns:
            A string representation of an object of this class.
        """
        return f"N( {self.mean}, {self.covar})"

    def plot(self, ax=None):
        if not ax:
            ax = plt.axes()
        return plot_error_ellipse(self.mean[:2], self.covar[:2, :2], ax)
