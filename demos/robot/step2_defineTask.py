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
import os
from pathlib import Path

from TaskThrowBall import TaskThrowBall
from dmpbbo.DmpBboJSONEncoder import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="directory to write task to")
    parser.add_argument("filename", help="file to write task to")
    args = parser.parse_args()

    x_goal = -0.70
    x_margin = 0.01
    y_floor = -0.3
    acceleration_weight = 0.001
    task = TaskThrowBall(x_goal, x_margin, y_floor, acceleration_weight)

    # Save the task instance itself
    os.makedirs(args.directory, exist_ok=True)
    filename = Path(args.directory, args.filename)
    print(f"  * Saving task to file {filename}")
    json = jsonpickle.encode(task)
    with open(filename, "w") as text_file:
        text_file.write(json)
