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
"""Script for dmp demo."""

import numpy as np
from matplotlib import pyplot as plt

from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dmps.DmpContextualTwoStep import DmpContextualTwoStep
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN
from dmpbbo.functionapproximators.FunctionApproximatorWLS import FunctionApproximatorWLS


def main():
    """ Main function of the script. """
    tau = 0.5
    n_time_steps = 51
    ts = np.linspace(0, tau, n_time_steps)

    n_dims = 1
    y_init = np.linspace(0.0, 0.7, n_dims)
    y_attr = np.linspace(0.4, 0.5, n_dims)

    if n_dims == 2:
        y_yd_ydd_viapoint = np.array([-0.2, 0.4, 0.0, 0.0, 0.0, 0.0])
    else:
        y_yd_ydd_viapoint = np.array([-0.2, 0.0, 0.0])
    viapoint_time = 0.4 * ts[-1]

    params_train = np.linspace(-0.5, 0.1, 3)
    params_and_trajs = []
    for param in params_train:
        y_yd_ydd_viapoint[0] = param
        traj = Trajectory.from_viapoint_polynomial(ts, y_init, y_yd_ydd_viapoint, viapoint_time, y_attr)
        params_and_trajs.append((param, traj))

    dmp_type = "KULVICIUS_2012_JOINING"
    fas_dmp = [FunctionApproximatorRBFN(10, 0.7) for _ in range(n_dims)]
    fa_ppf = FunctionApproximatorWLS()
    dmp_contextual = DmpContextualTwoStep(
        params_and_trajs, fas_dmp, ["weights"], fa_ppf, dmp_type=dmp_type, save_training_data=True)
    h, axs = dmp_contextual.plot_training(ts)

    params_test = np.linspace(-0.6, 0.2, 10)
    for params in params_test:
        task_params = np.array([params])
        xs, xds, ft, fa = dmp_contextual.analytical_solution(task_params, ts)
        h, _ = dmp_contextual.plot(ts, xs, xds, forcing_terms=ft, fa_outputs=fa, axs=axs)
        plt.setp(h, linestyle="-", linewidth=1, color=(0.0, 0.5, 0.0))

    plt.show()


if __name__ == "__main__":
    main()
