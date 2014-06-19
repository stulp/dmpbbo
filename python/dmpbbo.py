from numpy import array, linspace

import _dmpbbo

class Dmp(object):
    def __init__(self, n_dims_dmp, n_basis_functions):
        self.delegate = _dmpbbo.Dmp(n_dims_dmp, n_basis_functions)
        self.n_dims = n_dims_dmp

    def trajectory(self, duration, n_steps, weights):
        xs, xds, xdds = self.delegate.trajectory(duration, n_steps, weights)
        return xs, xds, xdds

    def set_tau(self, tau):
        self.delegate.set_tau(tau)

    # def test(self, time_vect, weights):
    #     phase, target = self.delegate.test(time_vect, weights)
    #     return phase, target

class UpdaterCovarAdaptation(object):
    def __init__(self, eliteness, weighting_method, base_level, diag_only, learning_rate, init_mean, init_covar):
        self.delegate = _dmpbbo.UpdaterCovarAdaptation(eliteness, weighting_method, base_level, diag_only, learning_rate, init_mean, init_covar)

    def update_distribution(self, samples, costs):
        self.delegate.update_distribution(samples.tolist(), costs.tolist())

    @property
    def mean(self):
        return array(self.delegate.get_mean())

    @property
    def covariance(self):
        return array(self.delegate.get_covariance())
