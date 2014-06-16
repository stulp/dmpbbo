#! /usr/bin/env python

import sys
sys.path.append('../build_dir/python')

import dmpbbo

n_dims = 2
updater = dmpbbo.UpdaterCovarAdaptation(10, "PI-BB", [1e-5] * n_dims, False, 0.75, [0.] * n_dims, [[1.] * n_dims] * n_dims)
updater.update_distribution([[1, 1], [0, 1]], [1.0, 0.2])

print updater.mean
print updater.covariance

