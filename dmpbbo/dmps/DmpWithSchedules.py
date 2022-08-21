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
""" Module for the DMP with gain/force schedules class. """

import numpy as np

from dmpbbo.dmps.Dmp import Dmp


class DmpWithSchedules(Dmp):
    """
    Dynamical Movement Primitives with extra "schedules", e.g. gain or force schedules.

    The schedules are computed directly with a function approximator, and are not part of the
    dynamical system formulation.

    """

    def __init__(self, tau, y_init, y_attr, function_approximators, func_apps_schedules, **kwargs):
        super().__init__(tau, y_init, y_attr, function_approximators, **kwargs)

        self.min_schedules = kwargs.pop("min_schedules", None)
        self.max_schedules = kwargs.pop("max_schedules", None)

        self._func_apps_schedules = func_apps_schedules

        n_dims = len(self._func_apps_schedules)
        if np.isscalar(self.min_schedules):
            self.min_schedules = np.full(n_dims, self.min_schedules)
        if np.isscalar(self.max_schedules):
            self.max_schedules = np.full(n_dims, self.max_schedules)

    @classmethod
    def from_traj_sched(cls, trajectory, func_apps_dmp, func_apps_schedules, **kwargs):
        """Initialize a DMP with schedules by training it from a trajectory (with schedules)

        @param trajectory: the trajectory to train on
        @param func_apps_dmp: Function approximators for the forcing term of the Dmp
        @param func_apps_schedules: Function approximators for the schedules
        @param kwargs:
        - schedules: A matrix of (gain/force) schedules. If it is None, it is assumed the
        schedules are stored in the "misc" property of the trajectory.
        - min_schedules: min values for the schedules (scalar or array, one for each dimension)
        - max_schedules: max values per the schedules (scalar or array, one for each dimension)
        - All kwargs are eventually passed to the Dmp constructor.
        """

        # Relevant variables from trajectory
        tau = trajectory.ts[-1]
        y_init = trajectory.ys[0, :]
        y_attr = trajectory.ys[-1, :]

        dmp = cls(tau, y_init, y_attr, func_apps_dmp, func_apps_schedules, **kwargs)

        schedules = kwargs.pop("schedules", None)
        dmp.train_sched(trajectory, schedules)

        return dmp

    def dim_schedules(self):
        """ Get the dimensionality of the schedules.

        @return: Dimensionality of the schedules.
        """
        return len(self._func_apps_schedules)

    def train_sched(self, trajectory, schedules=None):
        """ Train the Dmp with schedules with a trajectory

        @param trajectory: The trajectory with which to train the DMP.
        @param schedules: A matrix of (gain/force) schedules. If it is None, it is assumed the
        schedules are stored in the "misc" property of the trajectory.
        """
        super().train(trajectory)

        xs, _, _, _ = self.analytical_solution()
        inputs = xs[:, self.PHASE]
        targets = schedules or trajectory.misc

        if targets is None:
            raise ValueError(
                "targets is None. This means neither schedules nor trajectory.misc " "was available"
            )

        if targets.shape[1] != len(self._func_apps_schedules):
            raise ValueError(
                f"Targets must have at least as many dimensions as there are "
                f"function approximators for the schedules. But {targets.shape[1]} != "
                f"{len(self._func_apps_schedules)} "
            )

        for i_dim in range(len(self._func_apps_schedules)):
            self._func_apps_schedules[i_dim].train(inputs, targets[:, i_dim])

    def analytical_solution_sched(self, ts=None):
        """Return analytical solution of the system at certain times

        @param ts: A vector of times for which to compute the analytical solutions.
            If None is passed, the ts vector from the trajectory used to train the DMP is used.
        @return: xs, xds, schedules: Sequence of state vectors and their rates of change. T x D
        where T is the number of times (the length of 'ts'), and D the size of the
        state (i.e. dim_x()). Schedules contains the extra gain/force schedules.
        """
        if ts is None:
            if self._ts_train is None:
                raise ValueError(
                    "Neither the argument 'ts' nor the member variable self._ts_train was set."
                )
            else:
                ts = self._ts_train  # Set the times to the ones the Dmp was trained on.

        n_time_steps = ts.size
        xs, xds, forcing_terms, fa_outputs = super().analytical_solution(ts)

        schedules = np.ndarray((n_time_steps, len(self._func_apps_schedules)))
        for i_dim in range(len(self._func_apps_schedules)):
            schedules[:, i_dim] = self._func_apps_schedules[i_dim].predict(xs[:, self.PHASE])
        schedules = self._enforce_schedule_bounds(schedules)

        return xs, xds, schedules, forcing_terms, fa_outputs

    def _enforce_schedule_bounds(self, schedules):
        if self.min_schedules is None and self.max_schedules is None:
            # No bounds, simply return schedules
            return schedules
        if schedules.ndim == 1:
            schedules = schedules.reshape((schedules.shape[0], 1))
        for i_dim in range(len(self._func_apps_schedules)):
            min_sched = self.min_schedules[i_dim] if self.min_schedules else None
            max_sched = self.max_schedules[i_dim] if self.max_schedules else None
            schedules[:, i_dim] = np.clip(schedules[:, i_dim], min_sched, max_sched)
        return schedules

    def integrate_start_sched(self, y_init=None):
        """ Start integrating the DMP with schedules with a new initial state.

        @param y_init: The initial state vector (y part)
        @return: x, xd , schedules - The first vector of state variables and their rates of
        change, as well as the first values for the schedules.
        """
        xs, xds = super().integrate_start(y_init)

        n_schedules = len(self._func_apps_schedules)
        schedules = np.ndarray((n_schedules,))
        for i_dim in range(n_schedules):
            schedules[i_dim] = self._func_apps_schedules[i_dim].predict(xs[self.PHASE])
        schedules = self._enforce_schedule_bounds(schedules)

        return xs, xds, schedules

    def integrate_step_sched(self, dt, x):
        """ Integrate the system one time step.

        @param dt: Duration of the time step
        @param x: Current state
        @return: x_updated, xd_updated, schedules - Updated state and its rate of change,
        as well as the schedules, dt time later
        """
        x, xd = super().integrate_step(dt, x)

        schedules = np.ndarray((len(self._func_apps_schedules),))
        for i_dim in range(len(self._func_apps_schedules)):
            schedules[i_dim] = self._func_apps_schedules[i_dim].predict(x[self.PHASE])
        schedules = self._enforce_schedule_bounds(schedules)

        return x, xd, schedules

    def states_as_trajectory_sched(self, ts, x_in, xd_in, schedules):
        """Get the output of a DMP dynamical system as a trajectory, including schedules.

        As it is a dynamical system, the state vector of a DMP contains the output of the goal,
        spring, phase and gating system. What we are most interested in is the output of the
        spring system. This function extracts that information, and also computes the
        accelerations of the spring system, which are only stored implicitly in xd_in because
        second order systems are converted to first order systems with expanded state.

        The schedules are added as misc variables to the trajectory.

        @param ts: A vector of times
        @param x_in: State vector over time
        @param xd_in: State vector over time (rates of change)
        @param schedules: Gain/force schedules over time

        Return:
            Trajectory representation of the DMP state vector output.
        """
        traj = super().states_as_trajectory(ts, x_in, xd_in)
        traj.misc = schedules
        return traj

    def set_selected_param_names(self, names):
        """ Get the names of the parameters that have been selected.

        @param names: Names of the parameters that have been selected.
        """
        if isinstance(names, str):
            names = [names]  # Convert to list

        # Pass all names starting with "sched_" to the function approximators
        # (but without the prefix "sched_")
        sched_names = [n[len("sched_") :] for n in names if "sched_" in n]
        for fa in self._func_apps_schedules:
            fa.set_selected_param_names(sched_names)

        # Pass all names not starting with "sched_" to the Dmp superclass
        other_names = [n for n in names if "sched_" not in n]
        super().set_selected_param_names(other_names)

    def get_param_vector(self):
        """Get a vector containing the values of the selected parameters."""
        values = super().get_param_vector()
        for fa in self._func_apps_schedules:
            if fa.is_trained():
                values = np.append(values, fa.get_param_vector())
        return values

    def set_param_vector(self, values):
        """Set a vector containing the values of the selected parameters."""
        size = self.get_param_vector_size()
        if len(values) != size:
            raise ValueError("values must have size {size}")

        # First, pass values to superclass Dmp
        size_super = super().get_param_vector_size()
        if size_super > 0:
            super().set_param_vector(values[:size_super])

        # Second, set values for local func_apps for the schedules
        values_schedules = values[size_super:]
        offset = 0
        for fa in self._func_apps_schedules:
            if fa.is_trained():
                cur_size = fa.get_param_vector_size()
                cur_values = values_schedules[offset : offset + cur_size]
                fa.set_param_vector(cur_values)
                offset += cur_size

    def get_param_vector_size(self):
        """Get the size of the vector containing the values of the selected parameters."""
        size = self._get_param_vector_size_local()
        for fa in self._func_apps_schedules:
            if fa.is_trained():
                size += fa.get_param_vector_size()
        return size

    def plot_sched(self, ts, xs, xds, schedules, **kwargs):
        """ Plot the output of the DMP.

        @param ts: Time steps
        @param xs: States
        @param xds: Derivative of states
        @param schedules: Values of the schedules over time

        @return: The axes on which the plots were made.
        """
        lines, axs = super().plot(ts, xs, xds, **kwargs)

        ax = axs[4]
        ax.plot(ts, schedules)
        x = np.mean(ax.get_xlim())
        y = np.mean(ax.get_ylim())
        ax.text(x, y, "schedules", horizontalalignment="center")
        ax.set_xlabel(r"time ($s$)")
        ax.set_ylabel(r"schedules")
        return lines, axs
