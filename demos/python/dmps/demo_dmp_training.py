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
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
""" Script for training a DMP from a trajectory. """

import dmpbbo.json_for_cpp as json_for_cpp
from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN


def main():
    """ Main function for script. """

    # Train a DMP with a trajectory
    traj = Trajectory.loadtxt("trajectory.txt")
    function_apps = [FunctionApproximatorRBFN(10, 0.7) for _ in range(traj.dim)]
    dmp = Dmp.from_traj(traj, function_apps, dmp_type="KULVICIUS_2012_JOINING")

    # Compute analytical solution
    xs, xds, _, _ = dmp.analytical_solution(traj.ts)
    traj_reproduced = dmp.states_as_trajectory(traj.ts, xs, xds)  # noqa

    # Numerical integration
    dt = 0.001
    n_time_steps = int(1.3 * traj.duration / dt)
    x, xd = dmp.integrate_start()
    for tt in range(1, n_time_steps):
        x, xd = dmp.integrate_step(dt, x)
        # Convert complete DMP state to end-eff state
        y, yd, ydd = dmp.states_as_pos_vel_acc(x, xd)
        print(y)

    # Save the DMP to a json file that can be read in C++
    filename = "dmp_for_cpp.json"
    json_for_cpp.savejson_for_cpp(filename, dmp)
    print(f'Saved {filename} to local directory.')


if __name__ == "__main__":
    main()
