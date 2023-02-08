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
from matplotlib import cm

from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dmps.Trajectory import Trajectory
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

    def __init__(self, tau, y_init, y_attr, dmp_function_apps, **kwargs):
        """Initialize a contextual DMP

        @param dmp: DMP to be parameterized by the context
        @param ppf_function_approximator: Function approximator for the policy parameter function
        """
        dim_y = len(y_init)
        dim_x = 3 * dim_y + 2
        dim_x += y_init.size  # new: damping coefficient system
        super().__init__(1, tau, y_init, dim_x)

        self.dmp_function_apps = dmp_function_apps
        self.dmp = Dmp(tau, y_init, y_attr, dmp_function_apps, **kwargs)

        # The policy parameter function can only be trained once trajectories have
        # been provided.
        self.ppf = None
        self.task_params_train = None

    def train(self, task_params_and_trajs, param_names, ppf_function_app, **kwargs):
        save_training_data = kwargs.get("save_training_data", False)

        self.dmp.set_selected_param_names(param_names)

        # Train the policy parameter function
        targets = []  # The dmp parameters
        inputs = []  # The task parameters
        taus = []
        y_attrs = []
        y_inits = []
        cur_dmp = None
        for task_param_and_traj in task_params_and_trajs:
            task_params = np.atleast_1d(task_param_and_traj[0])
            inputs.append(task_params)

            traj = task_param_and_traj[1]
            cur_dmp = copy.deepcopy(self.dmp)
            cur_dmp.train(traj)
            # h, axs = cur_dmp.plot(plot_demonstration=traj)
            dmp_params = cur_dmp.get_param_vector()
            n_dmp_params = len(dmp_params)
            targets.append(dmp_params)

            taus.append(cur_dmp.tau)
            y_attrs.append(cur_dmp.y_attr)
            y_inits.append(cur_dmp.y_init)

        self.dmp = cur_dmp
        self.dmp.tau = np.mean(taus)
        self.dmp.y_init = np.mean(y_inits, axis=0)
        self.dmp.y_attr = np.mean(y_attrs, axis=0)

        targets = np.array(targets)
        inputs = np.array(inputs)
        # Useful for plotting and sanity checks
        self.task_params_train = inputs

        # ppf = policy parameter function
        n_dmp_params = 0
        if isinstance(ppf_function_app, list):
            if not len(ppf_function_app) == n_dmp_params:
                raise RuntimeError(
                    f"Length of 'ppf_function_app' list ({len(ppf_function_app)}) must be the as"
                    f" the vector of dmp parameters ({n_dmp_params}). "
                )
            self.ppf = ppf_function_app
        else:
            # Deep copies of separate function approximator, one for each DMP dim
            self.ppf = [copy.deepcopy(ppf_function_app) for _ in range(n_dmp_params)]

        for i_param in range(n_dmp_params):
            self.ppf[i_param].train(
                inputs, targets[:, i_param], save_training_data=save_training_data
            )

        self._task_params_and_trajs = task_params_and_trajs if save_training_data else None

    @classmethod
    def from_trajs(
        cls, task_params_and_trajs, dmp_function_apps, param_names, ppf_function_app, **kwargs
    ):
        """Initialize a DMP by training it from a trajectory.

        @param trajectories: the trajectories to train on
        @param dmp_function_apps: Function approximators for the DMP forcing term
        @param kwargs: All kwargs are passed to the Dmp constructor.
        """

        # Relevant variables from trajectory
        traj_index = 1
        first_traj = task_params_and_trajs[0][
            traj_index
        ]  # For now it is assumed tau, y_init and y_attr are the same for all
        tau = first_traj.ts[-1]
        y_init = first_traj.ys[0, :]
        y_attr = first_traj.ys[-1, :]

        dmp_contextual = cls(tau, y_init, y_attr, dmp_function_apps, **kwargs)

        dmp_contextual.train(task_params_and_trajs, param_names, ppf_function_app, **kwargs)

        return dmp_contextual

    @classmethod
    def from_dmp(cls, task_params_and_trajs, dmp, param_names, ppf_function_app, **kwargs):
        """Initialize a DMP by training it from a trajectory.

        @param trajectories: the trajectories to train on
        @param dmp: The DMP
        """

        # Relevant variables from trajectory

        dmp_contextual = cls(dmp.tau, dmp.y_init, dmp.y_attr, dmp._function_approximators)
        # Replace the dmp that was constructure in the above by the one that was passed.
        dmp_contextual.dmp = dmp

        dmp_contextual.train(task_params_and_trajs, param_names, ppf_function_app, **kwargs)

        return dmp_contextual

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

        # Predict dmp parameters from task parameters using the ppf
        n_params = self.dmp.get_param_vector_size()
        param_vector = np.zeros((n_params, 1))
        for i_param in range(n_params):
            param_vector[i_param] = self.ppf[i_param].predict(task_params)

        # Set the dmp parameters
        self.dmp.set_param_vector(param_vector)

        return self.dmp.integrate_start(y_init)

    def differential_equation(self, x):
        return self.dmp.differential_equation(x)

    def states_as_pos_vel_acc(self, x_in, xd_in):
        return self.dmp.states_as_pos_vel_acc(x_in, xd_in)

    def states_as_trajectory(self, ts, x_in, xd_in):
        return self.dmp.states_as_trajectory(ts, x_in, xd_in)

    def decouple_parameters(self):
        self.dmp.decouple_parameters()

    def plot_policy_parameter_function(self):
        n_rows = int(np.ceil(np.sqrt(len(self.ppf))))
        fig, axs = plt.subplots(n_rows, n_rows)
        axs = axs.flatten()
        for i_param, fa in enumerate(self.ppf):
            fa.plot(ax=axs[i_param], plot_model_parameters=True)

    def plot(self, task_params_and_trajs, **kwargs):

        axs = kwargs.get("axs", None)
        if axs is None:
            axs = self.dmp.get_dmp_axes()
            kwargs["axs"] = axs  # This will be passed to self.dmp later

        max_duration = 0.0
        for task_param_and_traj in task_params_and_trajs:
            traj_demo = task_param_and_traj[1]
            max_duration = max(max_duration, traj_demo.duration)
            dt = traj_demo.dt_mean

            h_demo, _ = traj_demo.plot(axs=axs[1:4])
            plt.setp(h_demo, linestyle="-", linewidth=3, color=(0.7, 0.7, 0.7))

        ts = np.arange(0.0, max_duration, dt)

        for task_param_and_traj in task_params_and_trajs:
            traj_demo = task_param_and_traj[1]
            task_param = task_param_and_traj[0]

            self.dmp.tau = traj_demo.duration
            self.dmp.y_init = traj_demo.y_init
            self.dmp.y_attr = traj_demo.y_final
            self.set_task_params(np.array([task_param]))

            h, _ = self.dmp.plot(ts, axs=axs, plot_no_forcing_term_also=True)
            # plt.setp(h, color='k', linestyle=':')

            for ax in axs:
                ax.axvline(self.dmp.tau, color="k", linewidth=1)

        return h, axs
        # Determine the range of the task parameters used or training.
        task_params_train_min = self.task_params_train.min(axis=0)
        task_params_train_max = self.task_params_train.max(axis=0)

        n_train = self.task_params_train.shape[0]
        # n_test = 2*n_train + 1
        n_test = n_train

        params_test = np.linspace(task_params_train_min, task_params_train_max, n_test)

        cmap = cm.copper

        h = []
        prev_tau = self.dmp.tau
        for i_params in range(len(params_test)):
            params = params_test[i_params]
            self.set_task_params(params)
            # if tau_demos:
            #    self.dmp.tau = tau_demos[i_params]
            h_dmp, _ = self.dmp.plot(ts, **kwargs)

            scaled = (params[0] - task_params_train_min[0]) / (
                task_params_train_max[0] - task_params_train_min[0]
            )
            scaled = np.clip(scaled, 0, 1)
            plt.setp(h_dmp, color=cmap(scaled))
            h.extend(h_dmp)

        self.dmp.tau = prev_tau

        return h, axs
