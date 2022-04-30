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

import os
import subprocess
from pathlib import Path


def execute_binary(executable_name, arguments, print_command=False):

    # This is a workaround to find the bin directory...
    # I did not find another way to make the test scripts compatible with pytest
    # AND calling the script on command line.

    cur_dir = "bin"
    for go_up in range(6):
        rel_executable_name = Path(cur_dir, executable_name)
        cur_dir = Path("..", cur_dir)

        if os.path.isfile(rel_executable_name):
            command = f"{rel_executable_name} {arguments}"
            if print_command:
                print(command)
            subprocess.call(command, shell=True)
            return

    raise ValueError(
        f"Executable '{executable_name}' does not exist in any bin directory. Please call 'make "
        f"install' in the build directory first. "
    )
