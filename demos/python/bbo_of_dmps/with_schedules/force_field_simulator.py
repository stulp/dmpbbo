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
"""Script to simuulate DMP integration with a force field.
Has been implemented in a separate file to facilitate debugging.
"""

import random

import numpy as np
from matplotlib import pyplot as plt


def perform_rollout(dmp_sched, integrate_time, n_time_steps, field_strength, field_max_time):
    """
    Perform a rollout with a force field
    @param dmp_sched:  The DMP to integrate
    @param integrate_time:  The time to integrate the DMP
    @param n_time_steps: The number of time steps to integrate the DMP
    @param field_strength: The strength of thw force field
    @param field_max_time: The time at which the (Gaussian) force field has its mode
    @return:
    """
    ts = np.linspace(0.0, integrate_time, n_time_steps)
    dt = ts[1]

    r = {"ts": ts}  # The rollout containing all relevant  numpy ndarrays
    for v in ["ys_des", "yds_des", "ydds_des", "schedules", "ys_cur", "ydds_cur", "yds_cur"]:
        r[v] = np.zeros([n_time_steps, dmp_sched.dim_y])
    r["fields"] = np.zeros([n_time_steps, 1])

    x_des, xd_des, sch = dmp_sched.integrate_start_sched()
    des = dmp_sched.states_as_pos_vel_acc(x_des, xd_des)
    for i, v in enumerate(["ys", "yds", "ydds"]):
        r[v + "_des"][0, :] = des[i]  # Output of integrate_start are values at t=0
        r[v + "_cur"][0, :] = r[v + "_des"][0, :]  # Current at t=0 is equal ot desired
    r["schedules"][0, :] = sch

    for tt in range(1, n_time_steps):

        x_des, xd_des, sch = dmp_sched.integrate_step_sched(dt, x_des)
        r["ys_des"][tt, :], r["yds_des"][tt, :], r["ydds_des"][
            tt, :
        ] = dmp_sched.states_as_pos_vel_acc(x_des, xd_des)

        # Compute error terms
        y_err = r["ys_cur"][tt - 1, :] - r["ys_des"][tt, :]
        yd_err = r["yds_cur"][tt - 1, :] - r["yds_des"][tt, :]

        # Force due to PD-controller
        r["schedules"][tt, :] = sch
        gain = sch
        r["ydds_cur"][tt, :] = r["ydds_des"][tt, :] - gain * y_err - np.sqrt(gain) * yd_err

        # Force due to force_field
        time = ts[tt]
        max_time = field_max_time
        w = np.sqrt(0.05 * max_time)
        r["fields"][tt, 0] = field_strength * np.exp(-0.5 * np.square(time - max_time) / (w * w))
        r["ydds_cur"][tt, :] += r["fields"][tt, 0]

        # Euler integration
        r["yds_cur"][tt, :] = r["yds_cur"][tt - 1, :] + dt * r["ydds_cur"][tt, :]
        r["ys_cur"][tt, :] = r["ys_cur"][tt - 1, :] + dt * r["yds_cur"][tt, :]

    # Compute reference trajectory without perturbation (already done above)
    # xs, xds, schedules, _, _ = dmp_sched.analytical_solution_sched(ts)
    # traj = dmp_sched.states_as_trajectory_sched(ts, xs, xds, schedules)
    # r["ys_des"] = traj.ys
    # r["yds_des"] = traj.yds
    # r["ydds_des"] = traj.ydds

    return r


def main_perform_rollout(field_strength, gains, axs):
    """
    Perform one rollout with a certain field strength and gain
    @param field_strength:  The strength of the force field (between 0 and 200 is appropriate)
    @param gains: The gains (between 10 and 2000 is appropriate)
    @param axs: The axes to plot on
    """
    from dmpbbo.dmps.DmpWithSchedules import DmpWithSchedules
    from dmpbbo.dmps.Trajectory import Trajectory
    from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN

    n_time_steps = 51
    tau = 1.0
    y_init = np.array([0.0])
    y_attr = np.array([1.0])
    n_dims = len(y_init)

    ts = np.linspace(0, tau, n_time_steps)
    traj = Trajectory.from_min_jerk(ts, y_init, y_attr)
    schedule = np.full((n_time_steps, n_dims), gains)
    traj.misc = schedule

    function_apps = [FunctionApproximatorRBFN(8, 0.95) for _ in range(n_dims)]
    function_apps_schedules = [FunctionApproximatorRBFN(7, 0.95) for _ in range(n_dims)]
    dmp_sched = DmpWithSchedules.from_traj_sched(traj, function_apps, function_apps_schedules)

    integrate_time = 1.3 * tau
    field_max_time = 0.3 * tau
    r = perform_rollout(dmp_sched, integrate_time, n_time_steps, field_strength, field_max_time)

    h = axs[0].plot(r["ts"], r["fields"])
    axs[0].set_ylabel("force field")
    color = h[0].get_color()

    axs[1].plot(r["ts"], r["schedules"], color=color)
    axs[1].set_ylabel("gains")

    axs[2].plot(r["ts"], r["ys_des"], "--", color=color, linewidth=2)
    axs[2].plot(r["ts"], r["ys_cur"], "-", color=color, linewidth=1)
    axs[2].set_ylabel("y")
    # for i, v in enumerate(["ys", "yds", "ydds"]):
    #    axs[i + 2].plot(r["ts"], r[v + "_des"], "-", color=color)
    #    axs[i + 2].plot(r["ts"], r[v + "_cur"], "--", color=color)
    #    axs[i + 2].set_ylabel(v)


def main():
    """
    Main function for this script
    """
    field_strengths = [-10.0, -5.0, 0.0, 5.0, 10.0]
    gains = [10.0, 50.0, 250.0]

    n_rows = len(gains)
    n_cols = 3
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    row = 0
    for gain in gains:
        axs = [fig.add_subplot(n_rows, n_cols, 1 + row * n_cols + sp) for sp in range(n_cols)]
        for field_strength in field_strengths:
            main_perform_rollout(field_strength, gain, axs)
            for i in range(n_cols):
                axs[i].set_xlabel("time")
            axs[0].set_ylim([1.1 * min(field_strengths), 1.1 * max(field_strengths)])
            axs[1].set_ylim([0, 1.1 * max(gains)])
            axs[2].set_ylim([-0.1, 1.5])
            for ax in axs:
                ax.set_xlim([0.0, 1.1])
        row += 1

    # n_rows = 1
    # fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    # axs = [fig.add_subplot(n_rows, n_cols, 1 + sp) for sp in range(n_cols)]
    # for i in range(10):
    #    gain = 500.0
    #    field_strength = random.randrange(-200, 200)
    #    main_perform_rollout(field_strength, gain, axs)

    plt.show()


if __name__ == "__main__":
    main()
