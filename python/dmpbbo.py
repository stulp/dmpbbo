# import numpy
# import sys
# import os

import _dmpbbo


class UpdaterCovarAdaptation(object):
    def __init__(self, eliteness, weighting_method, base_level, diag_only, learning_rate, init_mean, init_covar):
        self.delegate = _dmpbbo.UpdaterCovarAdaptation(eliteness, weighting_method, base_level, diag_only, learning_rate, init_mean, init_covar)

    def update_distribution(self, samples, costs):
        self.delegate.update_distribution(samples, costs)

    @property
    def mean(self):
        return self.delegate.get_mean()

    @property
    def covariance(self):
        return self.delegate.get_covariance()


class DmpBbo(object):
    def __init__(self, **kwargs):
        self._delegate = _dmpbbo.DmpBbo()

    def run(self, tau, n_time_steps, n_basis_functions, input_dim, intersection, save_dir):
        self._delegate.run(tau, n_time_steps, n_basis_functions, input_dim, intersection, save_dir)
