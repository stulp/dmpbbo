# This file is part of DmpBbo, a set of libraries and programs for the
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2022 Freek Stulp
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
import os, sys

# Include scripts for plotting
lib_path = os.path.abspath("../../../python/")
sys.path.append(lib_path)

from to_jsonpickle import *

from functionapproximators.FunctionApproximatorLWR import *
from functionapproximators.FunctionApproximatorRBFN import *


def getTrainingData(n_dims):

    if n_dims == 1:
        n_samples_per_dim = 25
        inputs = np.linspace(0.0, 2.0, n_samples_per_dim)
        targets = 3 * np.exp(-inputs) * np.sin(2 * np.square(inputs))

    else:
        n_samples_per_dim = [11, 9]  # Does not work yet; kept for future debugging.
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
    return (inputs, targets)


if __name__ == "__main__":
    """Run some training sessions and plot results."""

    for n_dims in [1, 2]:

        (inputs, targets) = getTrainingData(n_dims)

        for fa_name in ["RBFN", "LWR"]:

            n_bfs = 9 if n_dims == 1 else [5, 5]

            # Initialize function approximator
            if fa_name == "LWR":
                intersection = 0.5
                fa = FunctionApproximatorLWR(n_bfs, intersection)
            else:
                intersection = 0.7
                fa = FunctionApproximatorRBFN(n_bfs, intersection)

            # Train function approximator with data
            fa.train(inputs, targets)

            s = to_jsonpickle(fa)

            # Save to file
            with open(f"{fa_name}_{n_dims}D.json", "w") as text_file:
                text_file.write(s)
