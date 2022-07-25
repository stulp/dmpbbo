# This file is part of DmpBbo, a set of libraries and programs for the
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2018 Freek Stulp
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
"""Script for trajectory demo."""

import numpy as np
from matplotlib import pyplot as plt

from dmpbbo.dmps.Trajectory import Trajectory


def main():
    """ Main function of the script. """
    ts = np.linspace(0, 0.5, 101)
    y_first = np.array([0.0, 1.0])
    yd_first = np.array([0.0, 0.0])
    ydd_first = np.array([0.0, 0.0])
    y_last = np.array([0.4, 0.5])
    yd_last = np.array([10.0, 0])
    ydd_last = np.array([0, 0])

    traj_min_jerk = Trajectory.from_min_jerk(ts, y_first, y_last)
    traj_min_jerk.plot()
    plt.gcf().canvas.set_window_title("min-jerk trajectory")

    traj = Trajectory.from_polynomial(ts, y_first, yd_first, ydd_first, y_last, yd_last, ydd_last)
    traj.plot()
    plt.gcf().canvas.set_window_title("polynomial trajectory")

    y_yd_ydd_viapoint = np.array([-0.2, 0.3, 0.0, 0.0, 0, 0])
    viapoint_time = 0.4 * ts[-1]
    traj = Trajectory.from_viapoint_polynomial(
        ts, y_first, y_yd_ydd_viapoint, viapoint_time, y_last
    )
    fig = plt.figure(figsize=(15, 4))
    axs = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
    traj.plot(axs)
    axs[0].plot(viapoint_time, y_yd_ydd_viapoint[0], "ok")
    axs[0].plot(viapoint_time, y_yd_ydd_viapoint[1], "ok")
    plt.gcf().canvas.set_window_title("polynomial viapoint trajectory")

    # Do low-pass filtering
    # Make a noisy trajectory
    y_noisy = traj_min_jerk.ys + 0.001 * np.random.random_sample(traj_min_jerk.ys.shape)
    traj = Trajectory(ts, y_noisy)
    # Plot it
    lines, axs = traj.plot()
    plt.setp(lines, linestyle="-", linewidth=1, color=(0.7, 0.7, 1.0))
    plt.setp(lines[0], label="noisy trajectory")

    cutoff = 10.0
    order = 3
    traj.apply_low_pass_filter(cutoff, order)
    lines, _ = traj.plot(axs)
    plt.setp(lines, linestyle="-", linewidth=2, color=(0.2, 0.8, 0.2))
    plt.setp(lines[0], label="filtered trajectory")
    axs[0].legend()

    plt.gcf().canvas.set_window_title("low-pass filtering")

    plt.show()


if __name__ == "__main__":
    main()
