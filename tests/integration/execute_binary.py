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


def execute_binary(executable_name, arguments, print_command=False):
    if not os.path.isfile(executable_name):
        raise ValueError(
            f"Executable '{executable_name}' does not exist. Please call 'make install' in the build directory first."
        )

    command = f"{executable_name} {arguments}"
    if print_command:
        print(command)

    subprocess.call(command, shell=True)
