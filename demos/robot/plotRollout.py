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

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np

import dmpbbo.DmpBboJSONEncoder as dj


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="file (txt) to read cost vars from")
    parser.add_argument("task", help="file (json) with task")
    args = parser.parse_args()

    cost_vars = np.loadtxt(args.filename)
    task = None
    if args.task is not None:
        with open(args.task, "r") as f:
            task = jsonpickle.decode(f.read())

    fig = plt.figure(1)
    n_subplots = 1
    ax = fig.add_subplot(1, n_subplots, n_subplots)
    task.plot_rollout(cost_vars, ax)

    filename = "plotRollout.png"
    print(f"Saving to file: {filename}")
    fig.savefig(filename)

    plt.show()
