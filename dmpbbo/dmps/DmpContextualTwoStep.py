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
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
#
""" Module for the DMP class. """
import copy

import matplotlib.pyplot as plt
import numpy as np

from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dynamicalsystems.DynamicalSystem import DynamicalSystem
from dmpbbo.functionapproximators.Parameterizable import Parameterizable


class DmpContextualTwoStep(DynamicalSystem):
    """ Contextual Dynamical Movement Primitive

    Uses the two-step approach, i.e.
        1. determine DMP parameters from context parameters before integration
        2. integrate DMP with those parameters

    See the paper "Learning compact parameterized skills with a single regression" and
    ""Learning parameterized skills" for the nomenclature:
        https://ieeexplore.ieee.org/abstract/document/7030008/
        https://arxiv.org/abs/1206.6398
    """

    def __init__(self, task_params_and_trajs, dmp_function_apps, param_names, ppf_function_app, **kwargs):
        """Initialize a contextual DMP

        @param dmp: DMP to be parameterized by the context
        @param ppf_function_approximator: Function approximator for the policy parameter function
        """
        first_traj = task_params_and_trajs[0][1]
        tau = first_traj.duration
        y_init = first_traj.y_init
        dim_y = first_traj.dim
        dim_x = 3 * dim_y + 2
        super().__init__(1, tau, y_init, dim_x)

        self.dmp = Dmp.from_traj(first_traj, dmp_function_apps, **kwargs)
        self.dmp.set_selected_param_names(param_names)
        n_params = self.dmp.get_param_vector_size()

        # ppf = policy parameter function
        if isinstance(ppf_function_app, list):
            self.ppf = ppf_function_app
        else:
            # Deep copies of separate function approximator for each DMP dim
            self.ppf = [copy.deepcopy(ppf_function_app) for _ in range(n_params) ]

        self.train(task_params_and_trajs, **kwargs)

    def train(self, task_params_and_trajs, **kwargs):
        # Train the policy parameter function
        targets = []  # The dmp parameters
        inputs = []  # The task parameters
        for task_param_and_traj in task_params_and_trajs:
            task_params = np.atleast_1d(task_param_and_traj[0])
            inputs.append(task_params)

            traj = task_param_and_traj[1]
            self.dmp.train(traj)
            dmp_params = self.dmp.get_param_vector()
            targets.append(dmp_params)

        inputs = np.array(inputs)
        targets = np.array(targets)

        # ax = None
        for i_param in range(targets.shape[1]):
            self.ppf[i_param].train(inputs, targets[:, i_param])
            #_, ax = self.ppf[i_param].plot(inputs, ax=ax, plot_model_parameters=True)

        if kwargs.get("save_training_data", False):
            self._task_params_and_trajs = task_params_and_trajs
        else:
            self._task_params_and_trajs = None

    def dim_dmp(self):
        return self.dmp.dim_dmp

    def set_task_params(self, task_params):
        dmp_params = self.dmp.get_param_vector()  # Get vector of right size; Values do not matter.
        for i_param, fa in enumerate(self.ppf):
            dmp_params[i_param] = fa.predict(task_params)
        self.dmp.set_param_vector(dmp_params)

    def analytical_solution(self, task_params, ts=None, suppress_forcing_term=False):
        self.set_task_params(task_params)
        return self.dmp.analytical_solution(ts, suppress_forcing_term)

    def integrate_start(self, task_params, y_init=None):
        """ Start integrating the DMP with a new initial state.

        @param y_init: The initial state vector (y part)
        @return: x, xd - The first vector of state variables and their rates of change
        """

        n_params = self.dmp.get_param_vector_size()
        param_vector = np.array(n_params)
        for i_param in n_params:
            param_vector[i_param] = self.ppf[i_param].predict(task_params)

        self.dmp.set_param_vector(param_vector)

        return self.dmp.integrate_start(y_init)

    def differential_equation(self, x):
        return self.dmp.differential_equation(x)

    def states_as_pos_vel_acc(self, x_in, xd_in):
        return self.dmp.states_as_pos_vel_acc(x_in, xd_in)

    def states_as_trajectory(self, ts, x_in, xd_in):
        return self.dmp.states_as_trajectory(ts, x_in, xd_in)

    def plot(self, ts=None, xs=None, xds=None, **kwargs):
        return self.dmp.plot(ts, xs, xds, **kwargs)

    def plot_training(self, ts, **kwargs):
        if not self._task_params_and_trajs:
            print("Can only plot when 'save_training_data' was passed to constructor.")
            return [], []

        axs = kwargs.get("axs", self.dmp.get_dmp_axes())
        h = []
        for task_param_and_traj in self._task_params_and_trajs:
            traj_demo = task_param_and_traj[1]
            h_demo, _ = traj_demo.plot(axs=axs[1:4])
            plt.setp(h_demo, linestyle="-", linewidth=4, color=(0.8, 0.8, 0.8))

            task_params = np.atleast_1d(task_param_and_traj[0])
            self.set_task_params(task_params)
            h_dmp, axs = self.dmp.plot(ts, axs=axs)
            h.extend(h_dmp)

        plt.setp(h, linestyle="--", linewidth=2, color=(0.0, 0.0, 0.5))
        return h, axs

