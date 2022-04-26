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

import argparse
import inspect
import os
import sys

import jsonpickle

from dmpbbo.bbo.DistributionGaussian import DistributionGaussian
from dmpbbo.bbo.updaters import *
from dmpbbo.dmp_bbo.run_one_update import *
from dmpbbo.DmpBboJSONEncoder import *
from TaskThrowBall import TaskThrowBall

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="directory to write results to")
    args = parser.parse_args()

    filename = os.path.join(args.directory, "task.json")
    with open(filename, "r") as f:
        task = jsonpickle.decode(f.read())

    filename = os.path.join(args.directory, "distribution_initial.json")
    with open(filename, "r") as f:
        distribution_init = jsonpickle.decode(f.read())

    filename = os.path.join(args.directory, "dmp_initial.json")
    with open(filename, "r") as f:
        dmp = jsonpickle.decode(f.read())

    n_samples_per_update = 5

    eliteness = 10
    weighting = "PI-BB"

    updater_mean = UpdaterMean(eliteness, weighting)

    decay = 0.85
    updater_decay = UpdaterCovarDecay(eliteness, weighting, decay)

    min_level = 0.3
    max_level = 3.0
    diag_only = False
    learning_rate = 0.5
    updater_cma = UpdaterCovarAdaptation(
        eliteness, weighting, max_level, min_level, diag_only, learning_rate
    )

    updater = updater_decay

    task_solver = None
    session = runOptimizationTaskPrepare(
        args.directory,
        task,
        task_solver,
        distribution_init,
        n_samples_per_update,
        updater,
        dmp,
    )
