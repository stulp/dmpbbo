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

import numpy as np


def leastSquares(inputs, targets, use_offset=True, regularization=0.0):
    """(Regularized) least squares with an optional bias. 
    Args:
        inputs: Input values. Size n_samples X n_input_dims
        targets: Target values. Size n_samples X n_ouput_dims
        weights: Weights, one for each sample. Size n_samples X 1
        use_offset: Use linear model "y = a*x + offset" instead of "y = a*x". Default: true.
        regularization: Regularization term for regularized least squares. Default: 0.0.
     Returns:
        Parameters of the linear model
    """
    weights = np.ones(inputs.shape[0])
    min_weight = 0.0
    return weightedLeastSquares(
        inputs, targets, weights, use_offset, regularization, min_weight
    )


def weightedLeastSquares(
    inputs, targets, weights, use_offset=True, regularization=0.0, min_weight=0.0
):
    """(Regularized) weighted least squares with an optional bias. 
    Args:
        inputs: Input values. Size n_samples X n_input_dims
        targets: Target values. Size n_samples X n_ouput_dims
        weights: Weights, one for each sample. Size n_samples X 1
        use_offset: Use linear model "y = a*x + offset" instead of "y = a*x". Default: true.
        regularization: Regularization term for regularized least squares. Default: 0.0.
        min_weight: Minimum weight taken into account for least squares. Samples with a lower weight are not included in the least squares regression. May lead to significant speed-up. Default: 0.0.
     Returns:
        Parameters of the linear model
    """

    # Inputs and targets should have "n_samples" rows. If not, transpose them.
    # if inputs.shape[0] != n_samples and inputs.shape[1] == n_samples:
    #    inputs = inputs.T
    # if targets.shape[0] != n_samples and targets.shape[1] == n_samples:
    #    targets = targets.T

    n_samples = weights.size

    # Make the design matrix
    if use_offset:
        # Add a column with 1s
        X = np.column_stack((inputs, np.ones(n_samples)))

    else:
        X = inputs

    if True:  # min_weight<=0.0: # Still need to implement this

        # Weights matrix
        W = np.diagflat(weights)

        # Regularization matrix
        n_input_dims = 1
        if X.ndim > 1:
            n_input_dims = X.shape[1]
        Gamma = regularization * np.identity(n_input_dims)

        if X.ndim == 1:
            # Otherwise matrix multiplication below will not work
            X = np.atleast_2d(X).T

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
        betas = np.dot(np.dot(np.dot(np.linalg.inv(to_invert), X.T), W), targets)
        return betas

    else:
        # Very low weights do not contribute to the line fitting
        # Therefore, we can delete the rows in W, X and targets for which W is small
        #
        # Example with min_weight = 0.1 (a very high value!! usually it will be lower)
        #    W =       [0.001 0.01 0.5 0.98 0.46 0.01 0.001]^T
        #    X =       [0.0   0.1  0.2 0.3  0.4  0.5  0.6 ;
        #               1.0   1.0  1.0 1.0  1.0  1.0  1.0  ]^T  (design matrix, so last column = 1)
        #    targets = [1.0   0.5  0.4 0.5  0.6  0.7  0.8  ]
        #
        # will reduce to
        #    W_sub =       [0.5 0.98 0.46 ]^T
        #    X_sub =       [0.2 0.3  0.4 ;
        #                   1.0 1.0  1.0  ]^T  (design matrix, last column = 1)
        #    targets_sub = [0.4 0.5  0.6  ]
        #
        # Why all this trouble? Because the submatrices will often be much
        # smaller than the full  ones, so they are much faster to invert (note the
        # .inverse() call)

        # // Get a vector where 1 represents that weights >= min_weight, and 0 otherswise
        # VectorXi large_enough = (weights.array() >= min_weight).select(VectorXi::Ones(weights.size()), VectorXi::Zero(weights.size()));
        #
        # // Number of samples in the submatrices
        # int n_samples_sub = large_enough.sum();
        #
        # // This would be a 1-liner in Matlab... but Eigen is not good with splicing.
        # VectorXd weights_sub(n_samples_sub);
        # MatrixXd X_sub(n_samples_sub,n_betas);
        # MatrixXd targets_sub(n_samples_sub,targets.cols());
        # int jj=0;
        # for (int ii=0; ii<n_samples; ii++)
        # {
        #  if (large_enough[ii]==1)
        #  {
        #    weights_sub[jj] = weights[ii];
        #    X_sub.row(jj) = X.row(ii);
        #    targets_sub.row(jj) = targets.row(ii);
        #    jj++;
        #  }
        # }
        #
        # // Do the same inversion as above, but with only a small subset of the data
        #
        # // Weights matrix
        # MatrixXd W_sub = weights_sub.asDiagonal();
        #
        # // Regularization matrix
        # MatrixXd Gamma = regularization*MatrixXd::Identity(n_betas,n_betas);
        #
        # // Least squares then is a one-liner
        # betas =  (X_sub.transpose()*W_sub*X_sub+Gamma).inverse()*X_sub.transpose()*W_sub*targets_sub;
        return betas


def linearPrediction(inputs, betas):
    """Predict the output of a line/plane/hyperplane.

    Args:
        inputs Input values. Size n_samples X n_input_dims
        betas Parameters of the linear model, i.e. y = betas[0]*x + betas[1]
    Returns:
        Predicted output values. Size n_samples X n_ouput_dims
    """

    inputs = inputs.reshape(inputs.shape[0], -1)
    n_input_dims = inputs.shape[1]

    n_beta = betas.size

    if n_input_dims == n_beta:
        # Apparently, not everybody has python3.5 installed, so don't use @
        # outputs = inputs@betas
        outputs = np.dot(inputs, betas)
    else:
        # There is an offset (AKA bias or intercept)
        if n_input_dims + 1 != n_beta:
            raise ValueError(
                f"betas is of the wrong size (is {n_beta}, should be {n_inputs_dims+1})"
            )
        # Apparently, not everybody has python3.5 installed, so don't use @
        # outputs = inputs@betas[0:n_beta-1] + betas[-1]
        outputs = np.dot(inputs, betas[0 : n_beta - 1]) + betas[-1]

    return outputs
