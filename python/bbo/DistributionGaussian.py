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


import numpy as np
import math
import os

class DistributionGaussian:
    """ \brief A class for representing a Gaussian distribution.
    """

    def __init__(self, mean=0.0, covar=1.0):
        """ Construct the Gaussian distribution with a mean and covariance matrix.
        \param[in] mean Mean of the distribution
        \param[in] covar Covariance matrix of the distribution
        """
        self.mean = mean
        self.covar = covar

    def generateSamples(self,n_samples=1):
        """ Generate samples from the distribution.
        \param[in] n_samples Number of samples to sample
        \return The samples themselves (size n_samples X dim(mean)
        """
        return np.random.multivariate_normal(self.mean, self.covar, n_samples)

    def maxEigenValue(self):
        """ Get the largest eigenvalue of the covariance matrix of this distribution.
        \return largest eigenvalue of the covariance matrix of this distribution.
        """
        return max(np.linalg.eigvals(self.covar))

    def __str__(self):
        """ Get a string representation of an object of this class.
        \return A string representation of an object of this class.
        """
        return 'N( '+str(self.mean)+', '+str(self.covar)+' )'