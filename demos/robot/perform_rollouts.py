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

import os
import subprocess
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import dmpbbo.json_for_cpp as jc
from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dmps.Trajectory import Trajectory


def execute_binary(executable_name, arguments, print_command=False):

    if not os.path.isfile(executable_name):
        raise ValueError(
            f"Executable '{executable_name}' does not exist. Please call 'make install' in the build "
            f"directory first."
        )

    command = f"{executable_name} {arguments}"
    if print_command:
        print(command)

    subprocess.call(command, shell=True)


def perform_rollouts(dmp, mode="python_simulation", directory="."):

    if mode == "python_simulation":
        return run_python_simulation(dmp)

    elif mode == "robot_executes_dmp":
        filename_dmp = Path(directory, f"dmp_rollout.json")
        print(f"Saving trained DMP to: {filename_dmp}")
        jc.savejson(Path(directory, f"dmp_rollout.json"), dmp)
        jc.savejson_for_cpp(Path(directory, f"dmp_rollout_  "), dmp)

        filename_cost_vars = Path(directory, "cost_vars.txt")
        arguments = f"{filename_dmp} {filename_cost_vars}"
        execute_binary("./robotExecuteDmp", arguments)

        cost_vars = np.loadtxt(str(filename_cost_vars))
        return cost_vars

    else:  # 'robot_executes_trajectory':

        dt = 0.01
        ts = np.arange(0, 1.5 * dmp.tau, dt)
        (xs, xds, forcing, fa_outputs) = dmp.analytical_solution(ts)
        traj = dmp.states_as_trajectory(ts, xs, xds)

        filename_traj = Path(directory, "robot_trajectory.txt")
        traj.savetxt(filename_traj)

        filename_cost_vars = Path(directory, "cost_vars.txt")
        arguments = f"{filename_traj} {filename_cost_vars}"
        execute_binary("./robotExecuteTrajectory", arguments)

        cost_vars = np.loadtxt(str(filename_cost_vars))
        return cost_vars


def run_python_simulation(dmp, y_floor=-0.3):

    dt = 0.01
    ts = np.arange(0, 1.5 * dmp.tau, dt)
    n_time_steps = len(ts)

    (x, xd) = dmp.integrate_start()

    # ts = cost_vars[:,0]
    # y = cost_vars[:,1:1+n_dims]
    # ydd = cost_vars[:,1+n_dims*2:1+n_dims*3]
    # ball = cost_vars[:,-2:]
    n_dims_y = dmp.dim_dmp()
    ys = np.zeros([n_time_steps, n_dims_y])
    yds = np.zeros([n_time_steps, n_dims_y])
    ydds = np.zeros([n_time_steps, n_dims_y])
    ys_ball = np.zeros([n_time_steps, n_dims_y])
    yd_ball = np.zeros([1, n_dims_y])
    ydd_ball = np.zeros([1, n_dims_y])

    (ys[0, :], yds[0, :], ydds[0, :]) = dmp.stateAsPosVelAcc(x, xd)
    ys_ball[0, :] = ys[0, :]

    ball_in_hand = True
    ball_in_air = False
    for ii in range(1, n_time_steps):
        (x, xd) = dmp.integrate_step(dt, x)
        (ys[ii, :], yds[ii, :], ydds[ii, :]) = dmp.stateAsPosVelAcc(x, xd)

        if ball_in_hand:
            # If the ball is in your hand, it moves along with your hand
            ys_ball[ii, :] = ys[ii, :]
            yd_ball = yds[ii, :]
            ydd_ball = ydds[ii, :]  # noqa

            if ts[ii] > 0.6:
                # Release the ball to throw it!
                ball_in_hand = False
                ball_in_air = True

        elif ball_in_air:
            # Ball is flying through the air
            ydd_ball[0] = 0.0  # No friction
            ydd_ball[1] = -9.81  # Gravity

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


def main():
    from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN

    from TaskThrowBall import TaskThrowBall

    filename_traj = "trajectory.txt"
    traj = Trajectory.loadtxt(filename_traj)

    n_bfs = 10
    n_dims = traj.dim()
    fas = [FunctionApproximatorRBFN(n_bfs, 0.7) for _ in range(n_dims)]
    dmp = Dmp.from_traj(traj, fas, dmp_type="KULVICIUS_2012_JOINING")

    modes = ["python_simulation", "robot_executes_trajectory", "robot_executes_dmp"]

    fig = plt.figure(1)
    axs = {mode: fig.add_subplot(311 + ii) for ii, mode in enumerate(modes)}

    for mode in modes:

        cost_vars = perform_rollouts(dmp, mode)

        x_goal = -0.70
        x_margin = 0.01
        y_floor = -0.3
        acceleration_weight = 0.001
        task = TaskThrowBall(x_goal, x_margin, y_floor, acceleration_weight)

        task.plot_rollout(cost_vars, axs[mode])
        axs[mode].set_title(mode)

    plt.show()


if __name__ == "__main__":
    main()
