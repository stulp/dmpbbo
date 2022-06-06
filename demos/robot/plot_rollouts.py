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
""" Script for plotting one rollout. """

import argparse
import os
from glob import glob
from pathlib import Path

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np


def main():
    """ Main function that is called when executing the script. """

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="file (txt) or directory to read cost vars from")
    parser.add_argument("task", help="file (json) with task")
    # parser.add_argument("--show", action="store_true", help="show result plots")
    parser.add_argument("--save", action="store_true", help="save result plots to png")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        directory = args.input
        cost_vars_filenames = glob(str(Path(directory, "*cost_vars*.txt")))
    else:
        cost_vars_filenames = [args.input]

    task = None
    if args.task is not None:
        with open(args.task, "r") as f:
            task = jsonpickle.decode(f.read())

    fig = plt.figure(1)
    fig.suptitle(args.input)
    n_subplots = 1
    ax = fig.add_subplot(1, n_subplots, n_subplots)
    for filename in cost_vars_filenames:
        cost_vars = np.loadtxt(filename)
        task.plot_rollout(cost_vars, ax)

    if args.save:
        filename = Path(directory, "plot_rollouts.png")
        print(f"Saving to file: {filename}")
        fig.savefig(filename)
    else:
        plt.show()


if __name__ == "__main__":
    main()
