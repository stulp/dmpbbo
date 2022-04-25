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

import numpy as np
import matplotlib.pyplot as plt
import os, sys, subprocess

lib_path = os.path.abspath("../../python/")
sys.path.append(lib_path)

from dmp.Dmp import *

y_floor = -0.3


def executeBinary(executable_name, arguments, print_command=False):

    if not os.path.isfile(executable_name):
        print("")
        print("ERROR: Executable '" + executable + "' does not exist.")
        print("Please call 'make install' in the build directory first.")
        print("")
        sys.exit(-1)

    command = executable_name + " " + arguments
    if print_command:
        print(command)

    subprocess.call(command, shell=True)


def performRollouts(dmp, mode="python_simulation", directory="."):

    if mode == "python_simulation":
        return run_python_simulation(dmp)

    elif mode == "robot_executes_dmp":
        filename_dmp = os.path.join(directory, f"dmp_rollout.json")
        print("Saving trained DMP to: " + filename_dmp)
        save_to_json_for_cpp_also = True
        saveToJSON(dmp, filename_dmp, save_to_json_for_cpp_also)

        filename_cost_vars = os.path.join(directory, "cost_vars.txt")
        arguments = f"{filename_dmp} {filename_cost_vars}"
        executeBinary("./robotExecuteDmp", arguments)

        cost_vars = np.loadtxt(filename_cost_vars)
        return cost_vars

    else:  # 'robot_executes_trajectory':

        dt = 0.01
        ts = np.arange(0, 1.5 * dmp.tau_, dt)
        (xs, xds, forcing, fa_outputs) = dmp.analyticalSolution(ts)
        traj = dmp.statesAsTrajectory(ts, xs, xds)

        filename_traj = os.path.join(directory, "robot_trajectory.txt")
        traj.saveToFile(".", filename_traj)

        filename_cost_vars = os.path.join(directory, "cost_vars.txt")
        arguments = f"{filename_traj} {filename_cost_vars}"
        executeBinary("./robotExecuteTrajectory", arguments)

        cost_vars = np.loadtxt(filename_cost_vars)
        return cost_vars


def run_python_simulation(dmp, y_floor=-0.3):

    dt = 0.01
    ts = np.arange(0, 1.5 * dmp.tau, dt)
    n_time_steps = len(ts)

    (x, xd) = dmp.integrateStart()

    # ts = cost_vars[:,0]
    # y = cost_vars[:,1:1+n_dims]
    # ydd = cost_vars[:,1+n_dims*2:1+n_dims*3]
    # ball = cost_vars[:,-2:]
    n_dims_y = dmp.dim_dmp()
    ys = np.zeros([n_time_steps, n_dims_y])
    yds = np.zeros([n_time_steps, n_dims_y])
    ydds = np.zeros([n_time_steps, n_dims_y])
    ys_ball = np.zeros([n_time_steps, n_dims_y])

    (ys[0, :], yds[0, :], ydds[0, :]) = dmp.stateAsPosVelAcc(x, xd)
    ys_ball[0, :] = ys[0, :]

    ball_in_hand = True
    ball_in_air = False
    for ii in range(1, n_time_steps):
        (x, xd) = dmp.integrateStep(dt, x)
        (ys[ii, :], yds[ii, :], ydds[ii, :]) = dmp.stateAsPosVelAcc(x, xd)

        if ball_in_hand:
            # If the ball is in your hand, it moves along with your hand
            ys_ball[ii, :] = ys[ii, :]
            yd_ball = yds[ii, :]
            ydd_ball = ydds[ii, :]

            if ts[ii] > 0.6:
                # Release the ball to throw it!
                ball_in_hand = False
                ball_in_air = True

        elif ball_in_air:
            # Ball is flying through the air
            ydd_ball = 0.0  # No friction
            ydd_ball = -9.81  # Gravity

            # Euler integration
            yd_ball = yd_ball + dt * ydd_ball
            ys_ball[ii, :] = ys_ball[ii - 1, :] + dt * yd_ball

            if ys_ball[ii, 1] < y_floor:
                # Ball hits the floor (floor is at -0.3)
                ball_in_air = False
                ys_ball[ii, 1] = y_floor

        else:
            # Ball on the floor: does not move anymore
            ys_ball[ii, :] = ys_ball[ii - 1, :]

    ts = np.atleast_2d(ts).T
    cost_vars = np.concatenate((ts, ys, yds, ydds, ys_ball), axis=1)
    return cost_vars


if __name__ == "__main__":
    from functionapproximators.FunctionApproximatorRBFN import *
    from functionapproximators.FunctionApproximatorLWR import *

    from TaskThrowBall import TaskThrowBall

    filename_traj = "trajectory.txt"
    traj = Trajectory.readFromFile(filename_traj)

    n_bfs = 10
    n_dims = traj.dim()
    fas = [FunctionApproximatorRBFN(n_bfs, 0.7) for i_dim in range(n_dims)]
    dmp = Dmp.from_traj(traj, fas, "Dmp", "KULVICIUS_2012_JOINING")

    modes = ["python_simulation", "robot_executes_trajectory", "robot_executes_dmp"]

    fig = plt.figure(1)
    axs = {mode: fig.add_subplot(311 + ii) for ii, mode in enumerate(modes)}

    for mode in modes:

        cost_vars = performRollouts(dmp, mode)

        x_goal = -0.70
        x_margin = 0.01
        acceleration_weight = 0.001
        task = TaskThrowBall(x_goal, x_margin, y_floor, acceleration_weight)

        task.plotRollout(cost_vars, axs[mode])
        axs[mode].set_title(mode)

    plt.show()
