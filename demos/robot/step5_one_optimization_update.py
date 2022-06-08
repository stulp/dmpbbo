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
""" Script for doing one optimization update. """


import argparse

from dmpbbo.bbo_of_dmps.LearningSessionTask import LearningSessionTask
from dmpbbo.bbo_of_dmps.run_one_update import run_optimization_task_one_update


def main():
    """ Main function that is called when executing the script. """

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="directory to write results to")
    parser.add_argument("update", type=int, help="update number")
    parser.add_argument("--traj", action="store_true", help="integrate DMP and save trajectory")
    args = parser.parse_args()

    session = LearningSessionTask.from_dir(args.directory)

    run_optimization_task_one_update(session, args.update, args.traj)


if __name__ == "__main__":
    main()
