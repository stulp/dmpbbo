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
""" Script for plotting the optimization. """

import argparse
from pathlib import Path

from matplotlib import pyplot as plt

from dmpbbo.bbo_of_dmps.LearningSessionTask import LearningSessionTask


def main():
    """ Main function that is called when executing the script. """

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="directory to read results from")
    # parser.add_argument("--show", action="store_true", help="show result plots")
    parser.add_argument("--save", action="store_true", help="save result plots to png")
    args = parser.parse_args()

    session = LearningSessionTask.from_dir(args.directory)
    session.plot()
    if args.save:
        plt.gcf().suptitle(args.directory)
        filename = Path(args.directory, "optimization.png")
        print(f"Saving png to: {filename}")
        plt.gcf().savefig(filename)

    plt.show()


if __name__ == "__main__":
    main()
