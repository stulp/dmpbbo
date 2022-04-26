# This file is part of DmpBbo, a set of libraries and programs for the
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2018, 2022 Freek Stulp
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


from dmpbbo.functionapproximators.BasisFunction import *
from dmpbbo.functionapproximators.FunctionApproximator import FunctionApproximator


class FunctionApproximatorWLS(FunctionApproximator):
    def __init__(self, use_offset=True, regularization=0.0):
        """Initialize a Least-Squares function approximator.

        Args:
            use_offset: Use linear model "y = a*x + offset" instead of "y = a*x". Default: true.
            regularization: Regularization term for regularized least squares. Default: 0.0.
        """
        meta_params = {"use_offset": use_offset, "regularization": regularization}

        model_param_names = ["slope"]
        if use_offset:
            model_param_names.append("offset")

        super().__init__(meta_params, model_param_names)

    @staticmethod
    def _train(inputs, targets, meta_params, **kwargs):

        use_offset = meta_params["use_offset"]
        regularization = meta_params["regularization"]

        n_samples = targets.size

        inputs = inputs.reshape(n_samples, -1)

        # Make the design matrix
        if use_offset:
            # Add a column with 1s
            X = np.column_stack((inputs, np.ones(n_samples)))
        else:
            X = inputs

        # Weights matrix
        weights = kwargs.get("weights", None)
        if weights is None:
            W = np.eye(n_samples)
        else:
            W = np.diagflat(weights)

        # Regularization matrix
        n_dims_X = X.shape[1]
        Gamma = regularization * np.identity(n_dims_X)

        # Compute beta
        # 1 x n_betas
        # = inv( (n_betas x n_sam)*(n_sam x n_sam)*(n_sam*n_betas) )*( (n_betas x n_sam)*(n_sam x n_sam)*(n_sam * 1) )
        # = inv(n_betas x n_betas)*(n_betas x 1)
        #
        # Least squares is a one-liner
        # Apparently, not everybody has python3.5 installed, so don't use @
        # betas = np.linalg.inv(X.T@W@X + Gamma)@X.T@W@targets
        # In python<=3.4, it is not a one-liner
        to_invert = np.dot(np.dot(X.T, W), X) + Gamma
        beta = np.dot(np.dot(np.dot(np.linalg.inv(to_invert), X.T), W), targets)

        if use_offset:
            model_params = {"slope": beta[:-1], "offset": beta[-1]}
        else:
            model_params = {"slope": beta}

        return model_params

    @staticmethod
    def _getActivations(inputs, model_params):
        return None

    @staticmethod
    def _predict(inputs, model_params):

        # Ensure n_dims=2, i.e. shape = (30,) => (30,1)
        inputs = inputs.reshape(inputs.shape[0], -1)

        slope = model_params["slope"]
        offset = model_params.get("offset", 0.0)

        # Apparently, not everybody has python3.5 installed, so don't use @
        # lines[:,i_line] =  inputs@slopes[i_line,:].T + offsets[i_line]
        outputs = np.dot(inputs, slope.T) + offset

        return outputs
