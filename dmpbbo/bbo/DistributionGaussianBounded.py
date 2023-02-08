# This file is part of DmpBbo, a set of libraries and programs for the
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2023 Freek Stulp, ENSTA-ParisTech
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
""" Module for the DistributionGaussianBounded class. """


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from dmpbbo.bbo.DistributionGaussian import DistributionGaussian


class DistributionGaussianBounded(DistributionGaussian):
    """ A class for representing a bounded Gaussian distribution.
    """

    def __init__(self, mean=0.0, covar=1.0, lower_bound=None, upper_bound=None):
        """ Construct the Gaussian distribution with a mean and covariance matrix.

        @param mean: Mean of the distribution
        @param covar: Covariance matrix of the distribution
        """
        super().__init__(mean, covar)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def generate_samples(self, n_samples=1):
        samples = super().generate_samples(n_samples)
        clipped_samples = samples.clip(self.lower_bound, self.upper_bound)
        return clipped_samples
