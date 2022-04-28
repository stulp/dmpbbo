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
import numpy as np
from matplotlib import pyplot as plt

from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import (
    FunctionApproximatorRBFN,
)

if __name__ == "__main__":

    tau = 0.5
    n_dims = 2
    n_time_steps = 51

    y_init = np.linspace(0.0, 0.7, n_dims)
    y_attr = np.linspace(0.4, 0.5, n_dims)

    ts = np.linspace(0, tau, n_time_steps)
    y_yd_ydd_viapoint = np.array([-0.2, 0.4, 0.0, 0.0, 0, 0])
    viapoint_time = 0.4 * ts[-1]
    traj = Trajectory.from_viapoint_polynomial(
        ts, y_init, y_yd_ydd_viapoint, viapoint_time, y_attr
    )

    function_apps = [FunctionApproximatorRBFN(10, 0.7) for i_dim in range(n_dims)]
    # dmp_type='IJSPEERT_2002_MOVEMENT'
    dmp_type = "KULVICIUS_2012_JOINING"
    # dmp_type='COUNTDOWN_2013'
    dmp_scaling = "AMPLITUDE_SCALING"
    dmp = Dmp.from_traj(traj, function_apps, dmp_type=dmp_type, forcing_term_scaling=dmp_scaling)

    tau_exec = 0.7
    n_time_steps = 71
    ts = np.linspace(0, tau_exec, n_time_steps)
    xs, xds, forcing, fas = dmp.analyticalSolution(ts)

    for selected_param_names in ["weights","goal",["weights","goal"]]:

        dmp.setSelectedParamNames(selected_param_names)
        values = dmp.getParamVector()

        # Plotting

        # Original Dmp
        h, axs = dmp.plotStatic(tau, ts, xs, xds, forcing_terms=forcing, fa_outputs=fas)
        plt.setp(h, color=[0.7, 0.7, 1.0], linewidth=6)

        # Perturbed DMPs
        for i_sample in range(5):

            rand_vector = 1.0 + 0.2 * np.random.standard_normal(values.shape)
            new_values = rand_vector * values
            dmp.setParamVector(new_values)

            xs, xds, forcing, fas = dmp.analyticalSolution(ts)
            h, _ = dmp.plotStatic(
                tau, ts, xs, xds, forcing_terms=forcing, fa_outputs=fas, axs=axs
            )
            plt.setp(h, color=[0.6, 0.0, 0.0], linewidth=1)

    plt.show()
