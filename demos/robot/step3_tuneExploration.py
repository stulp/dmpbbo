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


import argparse
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import dmpbbo.json_for_cpp as jc
from TaskThrowBall import TaskThrowBall
from dmpbbo.bbo.DistributionGaussian import DistributionGaussian
from perform_rollouts import perform_rollouts

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dmp", help="input dmp")
    parser.add_argument("output_directory", help="directory to write results to")
    parser.add_argument(
        "--sigma", help="sigma of covariance matrix", type=float, default=3.0
    )
    parser.add_argument("--nsamples", help="number of samples", type=int, default=12)
    parser.add_argument("--show", action="store_true", help="show result plots")
    args = parser.parse_args()

    sigma_dir = "sigma_%1.3f" % args.sigma
    directory = Path(args.output_directory, sigma_dir)

    filename = args.dmp
    print(f"Loading DMP from: {filename}")
    dmp = jc.loadjson(filename)

    ts = dmp._ts_train
    xs, xds, _, _ = dmp.analytical_solution(ts)
    traj_mean = dmp.states_as_trajectory(ts, xs, xds)

    fig = plt.figure(1)
    ax1 = fig.add_subplot(131)
    lines, _ = traj_mean.plot([ax1])
    plt.setp(lines, linewidth=3)

    parameter_vector = dmp.get_param_vector()

    n_samples = args.nsamples
    sigma = args.sigma
    covar_init = sigma * sigma * np.eye(parameter_vector.size)
    distribution = DistributionGaussian(parameter_vector, covar_init)

    filename = Path(directory, f"distribution.json")
    print(f"Saving sampling distribution to: {filename}")
    os.makedirs(directory, exist_ok=True)
    jc.savejson(filename, distribution)

    samples = distribution.generate_samples(n_samples)

    ax2 = fig.add_subplot(132)
    distribution.plot(ax2)
    ax2.plot(samples[:, 0], samples[:, 1], "o", color="#999999")

    ax3 = fig.add_subplot(133)

    y_floor = -0.3
    x_goal = -0.70
    x_margin = 0.01
    acceleration_weight = 0.001
    task = TaskThrowBall(x_goal, x_margin, y_floor, acceleration_weight)

    for i_sample in range(n_samples):

        dmp.set_param_vector(samples[i_sample, :])

        filename = Path(directory, f"dmp_sample_{i_sample}.json")
        print(f"Saving sampled DMP to: {filename}")
        jc.savejson(filename, dmp, save_for_cpp_also=True)

        (xs, xds, forcing, fa_outputs) = dmp.analytical_solution()
        traj_sample = dmp.states_as_trajectory(ts, xs, xds)
        lines, _ = traj_sample.plot([ax1])
        plt.setp(lines, color="#999999", alpha=0.5)

        cost_vars = perform_rollouts(dmp, "python_simulation", directory)

        task.plot_rollout(cost_vars, ax3)

    filename = "exploration.png"
    fig.savefig(Path(directory, filename))

    if args.show:
        plt.show()
