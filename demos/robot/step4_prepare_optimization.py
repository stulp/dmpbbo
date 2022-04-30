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
from pathlib import Path

import jsonpickle

import dmpbbo.bbo_for_dmps.run_one_update as run_one
from dmpbbo.bbo.updaters import UpdaterMean, UpdaterCovarDecay, UpdaterCovarAdaptation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="directory to write results to")
    args = parser.parse_args()

    filename = Path(args.directory, "task.json")
    with open(filename, "r") as f:
        task = jsonpickle.decode(f.read())

    filename = Path(args.directory, "distribution_initial.json")
    with open(filename, "r") as f:
        distribution_init = jsonpickle.decode(f.read())

    filename = Path(args.directory, "dmp_initial.json")
    with open(filename, "r") as f:
        dmp = jsonpickle.decode(f.read())

    n_samples_per_update = 5

    updater_name = "decay"
    if updater_name == "mean":
        updater = UpdaterMean(eliteness=10, weighting="PI-BB")
    elif updater_name == "decay":
        updater = UpdaterCovarDecay(eliteness=10, weighting="PI-BB", decay=0.85)
    else:
        updater = UpdaterCovarAdaptation(
            eliteness=10,
            weighting="PI-BB",
            max_level=3.0,
            min_level=0.3,
            diag_only=False,
            learning_rate=0.5,
        )

    task_solver = None
    run_one.run_optimization_task_prepare(
        args.directory,
        task,
        task_solver,
        distribution_init,
        n_samples_per_update,
        updater,
        dmp,
    )


if __name__ == "__main__":
    main()
