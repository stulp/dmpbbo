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


## \file demoDmpChangeGoal.py
## \author Freek Stulp
## \brief  Visualizes results of demoDmpChangeGoal.cpp
##
## \ingroup Demos
## \ingroup Dmps

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys

lib_path = os.path.abspath("../")
sys.path.append(lib_path)

lib_path = os.path.abspath("../../python")
sys.path.append(lib_path)

from functionapproximators.FunctionApproximatorLWR import *

from dmp.Dmp import *
from dmp.Trajectory import *
from dmp.dmp_plotting import *


def getDemoTrajectory(ts):

    use_viapoint_traj = True
    if use_viapoint_traj:
        n_dims = 2
        y_first = np.zeros(n_dims)
        y_last = 0.1 * np.ones(n_dims)
        if n_dims == 2:
            y_last[1] = -0.8
            viapoint_time = 0.25
            viapoint_location = -0.5 * np.ones(n_dims)
            if n_dims == 2:
                viapoint_location[1] = -0.8

        y_yd_ydd_viapoint = np.zeros(3 * n_dims)
        y_yd_ydd_viapoint[:n_dims] = viapoint_location
        return Trajectory.from_viapoint_polynomial(
            ts, y_first, y_yd_ydd_viapoint, viapoint_time, y_last
        )

    else:
        n_dims = 2
        y_first = np.linspace(0.0, 0.7, n_dims)  # Initial state
        y_last = np.linspace(0.4, 0.5, n_dims)  # Final state
        return Trajectory.from_min_jerk(ts, y_first, y_last)


if __name__ == "__main__":

    # GENERATE A TRAJECTORY
    tau = 0.5
    n_time_steps = 51
    ts = np.linspace(0, tau, n_time_steps)  # Time steps
    trajectory = getDemoTrajectory(ts)  # getDemoTrajectory() is implemented below
    y_init = trajectory.y_init
    y_attr = trajectory.y_final

    # fig = plt.figure(1)
    # axs1 = [ fig.add_subplot(231), fig.add_subplot(132), fig.add_subplot(133) ]
    # lines = plotTrajectory(trajectory.asMatrix(),axs1)
    # plt.show()

    n_dims = trajectory.dim()

    # WRITE THINGS TO FILE
    directory = "/tmp/demoDmpChangeGoalPython"
    trajectory.saveToFile(directory, "demonstration_traj.txt")

    scaling_vector = ["NO_SCALING", "G_MINUS_Y0_SCALING", "AMPLITUDE_SCALING"]

    for scaling in scaling_vector:

        # MAKE THE FUNCTION APPROXIMATORS
        function_apps = [FunctionApproximatorLWR(10), FunctionApproximatorLWR(10)]
        sigmoid_max_rate = -20
        dmp = Dmp(tau, y_init, y_attr, function_apps, "Dmp", sigmoid_max_rate, scaling)

        # CONSTRUCT AND TRAIN THE DMP
        directory_scaling = directory + "/" + scaling
        dmp.train(trajectory)

        tau_exec = 0.7
        n_time_steps = 71
        ts = np.linspace(0, tau_exec, n_time_steps)

        # INTEGRATE DMP TO GET REPRODUCED TRAJECTORY
        for goal_number in range(7):
            # 0 =>  1.5
            # 1 =>  1.0
            # 2 =>  0.5
            # 3 =>  0.0
            # 4 => -0.5
            # 5 => -1.0
            # 6 => -1.5
            y_attr_scaled = y_attr * (0.5 * (goal_number - 3))

            # ANALYTICAL SOLUTION
            dmp.set_attractor_state(y_attr_scaled)
            (xs, xds, forcing_terms, fa_outputs) = dmp.analyticalSolution(ts)
            traj_reproduced = dmp.statesAsTrajectory(ts, xs, xds)
            basename = "reproduced" + str(goal_number)
            traj_reproduced.saveToFile(directory_scaling, basename + "_traj.txt")

            dt = ts[1]
            xs_step = np.zeros([n_time_steps, dmp._dim_x])
            xds_step = np.zeros([n_time_steps, dmp._dim_x])

            dmp.set_attractor_state(y_attr_scaled)
            (x, xd) = dmp.integrateStart()
            xs_step[0, :] = x
            xds_step[0, :] = xd
            for tt in range(1, n_time_steps):
                (xs_step[tt, :], xds_step[tt, :]) = dmp.integrateStep(
                    dt, xs_step[tt - 1, :]
                )
            traj_reproduced = dmp.statesAsTrajectory(ts, xs_step, xds_step)
            basename = "reproduced_num" + str(goal_number)
            traj_reproduced.saveToFile(directory_scaling, basename + "_traj.txt")

    print("Plotting")

    for numerical_or_analytical in [1, 2]:
        figure_number = numerical_or_analytical
        fig = plt.figure(figure_number)
        if numerical_or_analytical == 1:
            fig.suptitle("Analytical Solution")
        else:
            fig.suptitle("Numerical Integration")
        axs1 = [fig.add_subplot(231), fig.add_subplot(232), fig.add_subplot(233)]
        axs2 = [fig.add_subplot(234), fig.add_subplot(235), fig.add_subplot(236)]

        trajectory = numpy.loadtxt(directory + "/demonstration_traj.txt")
        traj_dim0 = trajectory[:, [0, 1, 3, 5]]
        traj_dim1 = trajectory[:, [0, 2, 4, 6]]
        lines = plotTrajectory(traj_dim0, axs1)
        lines.extend(plotTrajectory(traj_dim1, axs2))
        plt.setp(lines, linestyle="-", linewidth=8, color=(0.4, 0.4, 0.4))

        for goal_number in range(7):

            scalings = ["NO_SCALING", "G_MINUS_Y0_SCALING", "AMPLITUDE_SCALING"]
            for scaling in scalings:

                directory_scaling = directory + "/" + scaling
                # print(directory_scaling)

                basename = "reproduced" + str(goal_number)
                if numerical_or_analytical == 2:
                    basename = "reproduced_num" + str(goal_number)
                trajectory = numpy.loadtxt(
                    directory_scaling + "/" + basename + "_traj.txt"
                )
                traj_dim0 = trajectory[:, [0, 1, 3, 5]]
                traj_dim1 = trajectory[:, [0, 2, 4, 6]]
                lines = plotTrajectory(traj_dim0, axs1)
                lines.extend(plotTrajectory(traj_dim1, axs2))

                if scaling == "NO_SCALING":
                    plt.setp(lines, linestyle="--", linewidth=4, color=(0.8, 0, 0))
                if scaling == "G_MINUS_Y0_SCALING":
                    plt.setp(lines, linestyle="--", linewidth=3, color=(0, 0.7, 0))
                if scaling == "AMPLITUDE_SCALING":
                    plt.setp(lines, linestyle="-", linewidth=1, color=(0, 0.7, 0.7))

        labels = ["Demonstration"]
        labels.extend(scalings)
        plt.legend(labels)

    plt.show()
