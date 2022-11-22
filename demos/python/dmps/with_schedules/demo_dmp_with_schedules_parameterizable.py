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
"""Script for dmp_parameterizable demo."""

import numpy as np
from matplotlib import pyplot as plt

from dmpbbo.dmps.DmpWithSchedules import DmpWithSchedules
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN


def main():
    """ Main function of the script. """
    tau = 0.5
    n_dims = 2
    n_time_steps = 51

    y_init = np.linspace(0.0, 0.7, n_dims)
    y_attr = np.linspace(0.4, 0.5, n_dims)

    ts = np.linspace(0, tau, n_time_steps)
    y_yd_ydd_viapoint = np.array([-0.2, 0.4, 0.0, 0.0, 0, 0])
    viapoint_time = 0.4 * ts[-1]
    traj = Trajectory.from_viapoint_polynomial(ts, y_init, y_yd_ydd_viapoint, viapoint_time, y_attr)

    sch_begin = np.linspace(5, 7, n_dims)
    sch_end = np.linspace(5, 7, n_dims)
    sch_viapoint = np.array([4, 8, 0.0, 0.0, 0, 0])
    sch_via_time = 0.5 * ts[-1]
    traj_schedule = Trajectory.from_viapoint_polynomial(
        ts, sch_begin, sch_viapoint, sch_via_time, sch_end
    )
    traj.misc = traj_schedule.ys

    function_apps = [FunctionApproximatorRBFN(10, 0.7) for _ in range(n_dims)]
    function_apps_schedules = [FunctionApproximatorRBFN(8, 0.95) for _ in range(n_dims)]
    dmp = DmpWithSchedules.from_traj_sched(traj, function_apps, function_apps_schedules)

    tau_exec = 0.7
    n_time_steps = 71
    ts = np.linspace(0, tau_exec, n_time_steps)
    xs, xds, scheds, forcing, fas = dmp.analytical_solution_sched(ts)

    for selected_param_names in ["weights", "sched_weights", ["weights", "sched_weights"]]:

        dmp.set_selected_param_names(selected_param_names)
        values = dmp.get_param_vector()

        # Plotting

        # Original Dmp
        h, axs = dmp.plot_sched(ts, xs, xds, scheds, forcing_terms=forcing, fa_outputs=fas)
        plt.setp(h, color=[0.7, 0.7, 1.0], linewidth=6)

        # Perturbed DMPs
        for i_sample in range(5):
            rand_vector = 1.0 + 0.05 * np.random.standard_normal(values.shape)
            new_values = rand_vector * values
            dmp.set_param_vector(new_values)

            xs, xds, scheds, forcing, fas = dmp.analytical_solution_sched(ts)
            h, _ = dmp.plot_sched(
                ts, xs, xds, scheds, forcing_terms=forcing, fa_outputs=fas, axs=axs
            )
            plt.setp(h, color=[0.6, 0.0, 0.0], linewidth=1)
        plt.gcf().canvas.set_window_title(f"Perturbation of  {selected_param_names}")

    plt.show()


if __name__ == "__main__":
    main()
