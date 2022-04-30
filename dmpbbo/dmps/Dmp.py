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

import matplotlib.pyplot as plt
import numpy as np

from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.dynamicalsystems.DynamicalSystem import DynamicalSystem
from dmpbbo.dynamicalsystems.ExponentialSystem import ExponentialSystem
from dmpbbo.dynamicalsystems.SigmoidSystem import SigmoidSystem
from dmpbbo.dynamicalsystems.SpringDamperSystem import SpringDamperSystem
from dmpbbo.dynamicalsystems.TimeSystem import TimeSystem
from dmpbbo.functionapproximators.Parameterizable import Parameterizable


class Dmp(DynamicalSystem, Parameterizable):
    def __init__(
        self,
        tau,
        y_init,
        y_attr,
        function_approximators,
        phase_system=None,  # Initialized later, depends on tau (https://peps.python.org/pep-0671/)
        gating_system=None,  # Initialized later, depends on tau (https://peps.python.org/pep-0671/)
        **kwargs,
    ):
        """Initialize a DMP with function approximators and subsystems

        Args:
            tau           : Time constant
            y_init        : Initial state
            y_attr        : Attractor state
            function_approximators : Function approximators for the forcing term
            forcing_term_scaling : Which method to use for scaling the forcing term
                ( "NO_SCALING", "G_MINUS_Y0_SCALING")
            alpha_spring_damper : alpha in the spring-damper system of the dmp
            goal_system   : Dynamical system to compute delayed goal
            phase_system  : Dynamical system to compute the phase
            gating_system : Dynamical system to compute the gating term
        """

        dim_dmp = 3 * y_init.size + 2
        super().__init__(1, tau, y_init, dim_dmp)

        sigmoid_max_rate = kwargs.get("sigmoid_max_rate", -15)
        forcing_term_scaling = kwargs.get("forcing_term_scaling", "NO_SCALING")
        alpha = kwargs.get("alpha_spring_damper", 20.0)
        goal_system = kwargs.get("goal_system", None)

        self._y_attr = y_attr

        self._function_approximators = function_approximators

        self._forcing_term_scaling = forcing_term_scaling
        self._scaling_amplitudes = None

        self._spring_system = SpringDamperSystem(tau, y_init, y_attr, alpha)

        # Set defaults for subsystems if necessary
        self._phase_system = phase_system or TimeSystem(tau)
        self._gating_system = gating_system or SigmoidSystem(
            tau, np.ones(1), sigmoid_max_rate, 0.85
        )
        self._goal_system = goal_system or ExponentialSystem(tau, y_init, y_attr, 15)

        self._ts_train = None

        self._selected_param_names = []

        d = self._dim_y
        self.SPRING = np.arange(0 * d + 0, 0 * d + 0 + 2 * d)
        self.SPRING_Y = np.arange(0 * d + 0, 0 * d + 0 + 1 * d)
        self.SPRING_Z = np.arange(1 * d + 0, 1 * d + 0 + 1 * d)
        self.GOAL = np.arange(2 * d + 0, 2 * d + 0 + 1 * d)
        self.PHASE = np.arange(3 * d + 0, 3 * d + 0 + 1)
        self.GATING = np.arange(3 * d + 1, 3 * d + 1 + 1)

    def dim_dmp(self):
        return self._dim_y

    @classmethod
    def from_traj(cls, trajectory, function_approximators, **kwargs):
        """Initialize a DMP by training it from a trajectory.

        Args:
            trajectory    - the trajectory to train on
            function_approximators - Function approximators for the forcing term
            dmp_type      - Type of the Dmp
                ( "IJSPEERT_2002_MOVEMENT", "KULVICIUS_2012_JOINING", "COUNTDOWN_2013")
            forcing_term_scaling - Which method to use for scaling the forcing term
                ( "NO_SCALING", "G_MINUS_Y0_SCALING", "AMPLITUDE_SCALING" )
            phase_system  - Dynamical system to compute the phase
            gating_system - Dynamical system to compute the gating term
        """

        dmp_type = kwargs.get("dmp_type", "KULVICIUS_2012_JOINING")
        forcing_term_scaling = kwargs.get("forcing_term_scaling", "AMPLITUDE_SCALING")

        # Relevant variables from trajectory
        tau = trajectory.ts[-1]
        y_init = trajectory.ys[0, :]
        y_attr = trajectory.ys[-1, :]

        # Initialize dynamical systems

        if dmp_type == "IJSPEERT_2002_MOVEMENT":
            goal_system = None
            phase_system = ExponentialSystem(tau, 1, 0, 4)
            gating_system = ExponentialSystem(tau, 1, 0, 4)

        elif dmp_type in ["KULVICIUS_2012_JOINING", "COUNTDOWN_2013"]:
            goal_system = ExponentialSystem(tau, y_init, y_attr, 15)
            sigmoid_max_rate = -15
            gating_system = SigmoidSystem(tau, 1, sigmoid_max_rate, 0.85)
            count_down = dmp_type == "COUNTDOWN_2013"
            phase_system = TimeSystem(tau, count_down)

        else:
            raise ValueError(f"Unknown dmp_type: {dmp_type}")

        alpha_spring_damper = 20.0
        dmp = cls(
            tau,
            y_init,
            y_attr,
            function_approximators,
            phase_system,
            gating_system,
            forcing_term_scaling=forcing_term_scaling,
            alpha_spring_damper=alpha_spring_damper,
            goal_system=goal_system,
        )

        dmp.train(trajectory)

        return dmp

    def integrate_start(self, y_init=None):
        """ Start integrating the DMP with a new initial state.

        Args:
            y_init - The initial state vector (y part)
        Returns:
            x, xd - The first vector of state variables and their rates of change
        """
        if y_init:
            self.y_init = y_init

        x = np.zeros(self._dim_x)
        xd = np.zeros(self._dim_x)

        # Start integrating goal system if it exists
        if self._goal_system is None:
            # No goal system, simply set goal state to attractor state
            x[self.GOAL] = self._y_attr
            xd[self.GOAL] = 0.0
        else:
            # Goal system exists. Start integrating it.
            (x[self.GOAL], xd[self.GOAL]) = self._goal_system.integrate_start()

        # Set the attractor state of the spring system
        self._spring_system.y_attr = x[self.GOAL]

        # Start integrating all further subsystems
        (x[self.SPRING], xd[self.SPRING]) = self._spring_system.integrate_start()
        (x[self.PHASE], xd[self.PHASE]) = self._phase_system.integrate_start()
        (x[self.GATING], xd[self.GATING]) = self._gating_system.integrate_start()

        # Add rates of change
        xd = self.differential_equation(x)
        return x, xd

    def differential_equation(self, x):
        """The differential equation which defines the system.

        It relates state values to rates of change of those state values

        Args:
            x - current state (column vector of size dim() X 1)

        Returns:
            Rate of change in state (column vector of size dim() X 1)
        """

        xd = np.zeros(x.shape)

        if self._goal_system is None:
            # If there is no dynamical system for the delayed goal, the goal is
            # simply the attractor state
            self._spring_system.y_attr = self._y_attr
            # with zero change
            xd[self.GOAL] = np.zeros(self._dim_y)
        else:
            # Integrate goal system and get current goal state
            self._goal_system.y_attr = self._y_attr
            x_goal = x[self.GOAL]
            xd[self.GOAL] = self._goal_system.differential_equation(x_goal)
            # The goal state is the attractor state of the spring-damper system
            self._spring_system.y_attr = x_goal

        # Integrate spring damper system
        # Forcing term is added to spring_state later
        xd[self.SPRING] = self._spring_system.differential_equation(x[self.SPRING])

        # Non-linear forcing term phase and gating systems
        xd[self.PHASE] = self._phase_system.differential_equation(x[self.PHASE])
        xd[self.GATING] = self._gating_system.differential_equation(x[self.GATING])

        fa_output = self._compute_func_approx_predictions(x[self.PHASE])

        # Gate the output of the function approximators
        gating = x[self.GATING]
        forcing_term = gating * fa_output

        # Scale the forcing term, if necessary
        if self._forcing_term_scaling == "G_MINUS_Y0_SCALING":
            g_minus_y0 = self._y_attr - self._y_init
            forcing_term = forcing_term * g_minus_y0

        elif self._forcing_term_scaling == "AMPLITUDE_SCALING":
            if self._scaling_amplitudes is None:
                raise ValueError("Cannot do AMPLITUDE_SCALING if not trained with trajectory.")
            forcing_term = forcing_term * self._scaling_amplitudes

        # Add forcing term to the ZD component of the spring state
        xd[self.SPRING_Z] += np.squeeze(forcing_term) / self._tau

        return xd

    def _compute_func_approx_predictions(self, phase_state):
        """Compute the outputs of the function approximators.

        Args:
            phase_state The phase states for which the outputs are computed.

        Returns:
            The outputs of the function approximators.
        """
        n_time_steps = phase_state.size
        fa_output = np.zeros([n_time_steps, self.dim_dmp()])

        for i_fa in range(self.dim_dmp()):
            if self._function_approximators[i_fa].is_trained():
                fa_output[:, i_fa] = self._function_approximators[i_fa].predict(phase_state)
        return fa_output

    def analytical_solution(self, ts=None):
        """Return analytical solution of the system at certain times

        Args:
            ts: A vector of times for which to compute the analytical solutions.
            If None is passed, the ts vector from the trajectory used to train the DMP is used.

        Returns:
            xs: Sequence of state vectors. T x D or D x T matrix, where T is the number of times
            (the length of 'ts'), and D the size of the state (i.e. dim())
            xds: Sequence of state vectors (rates of change). T x D or D x T matrix, where T is
            the number of times (the length of 'ts'), and D the size of the state (i.e. dim())

        The output xs and xds will be of size D x T only if the matrix x you pass as an argument
        of size D x T. In all other cases (i.e. including passing an empty matrix) the size of x
        will be T x D. This feature has been added so that you may pass matrices of either size.
        """
        if ts is None:
            if self._ts_train is None:
                raise ValueError(
                    "Neither the argument 'ts' nor the member variable self._ts_train was set."
                )
            else:
                ts = self._ts_train  # Set the times to the ones the Dmp was trained on.

        n_time_steps = ts.size

        # INTEGRATE SYSTEMS ANALYTICALLY AS MUCH AS POSSIBLE

        # Integrate phase
        (xs_phase, xds_phase) = self._phase_system.analytical_solution(ts)

        # Compute gating term
        (xs_gating, xds_gating) = self._gating_system.analytical_solution(ts)

        # Compute the output of the function approximator
        fa_outputs = self._compute_func_approx_predictions(xs_phase)

        # Gate the output to get the forcing term
        forcing_terms = fa_outputs * xs_gating

        # Scale the forcing term, if necessary
        if self._forcing_term_scaling == "G_MINUS_Y0_SCALING":
            g_minus_y0 = self._y_attr - self._y_init
            g_minus_y0_rep = np.tile(g_minus_y0, (n_time_steps, 1))
            forcing_terms *= g_minus_y0_rep

        elif self._forcing_term_scaling == "AMPLITUDE_SCALING":
            _scaling_amplitudes_rep = np.tile(self._scaling_amplitudes, (n_time_steps, 1))
            forcing_terms *= _scaling_amplitudes_rep

        # Get current delayed goal
        if self._goal_system is None:
            # If there is no dynamical system for the delayed goal, the goal is
            # simply the attractor state
            xs_goal = np.tile(self._y_attr, (n_time_steps, 1))
            # with zero change
            xds_goal = np.zeros(xs_goal.shape)
        else:
            # Integrate goal system and get current goal state
            xs_goal, xds_goal = self._goal_system.analytical_solution(ts)

        xs = np.zeros([n_time_steps, self._dim_x])
        xds = np.zeros([n_time_steps, self._dim_x])

        xs[:, self.GOAL] = xs_goal
        xds[:, self.GOAL] = xds_goal
        xs[:, self.PHASE] = xs_phase
        xds[:, self.PHASE] = xds_phase
        xs[:, self.GATING] = xs_gating
        xds[:, self.GATING] = xds_gating

        # THE REST CANNOT BE DONE ANALYTICALLY

        # Reset the dynamical system, and get the first state
        damping = self._spring_system.damping_coefficient
        local_spring_system = SpringDamperSystem(self._tau, self.y_init, self._y_attr, damping)

        # Set first attractor state
        local_spring_system.y_attr = xs_goal[0, :]

        # Start integrating spring damper system
        (x_spring, xd_spring) = local_spring_system.integrate_start()

        # For convenience
        SPRING = self.SPRING  # noqa
        SPRING_Y = self.SPRING_Y  # noqa
        SPRING_Z = self.SPRING_Z  # noqa

        t0 = 0
        xs[t0, SPRING] = x_spring
        xds[t0, SPRING] = xd_spring

        # Add forcing term to the acceleration of the spring state
        xds[0, SPRING_Z] = xds[0, SPRING_Z] + forcing_terms[t0, :] / self._tau

        for tt in range(1, n_time_steps):
            dt = ts[tt] - ts[tt - 1]

            # Euler integration
            xs[tt, SPRING] = xs[tt - 1, SPRING] + dt * xds[tt - 1, SPRING]

            # Set the attractor state of the spring system
            local_spring_system.y_attr = xs[tt, self.GOAL]

            # Integrate spring damper system
            xds[tt, SPRING] = local_spring_system.differential_equation(xs[tt, SPRING])

            # Add forcing term to the acceleration of the spring state
            xds[tt, SPRING_Z] = (
                xds[tt, SPRING_Z] + forcing_terms[tt, :] / self._tau
            )  # + perturbation
            # Compute y component from z
            xds[tt, SPRING_Y] = xs[tt, SPRING_Z] / self._tau

        return xs, xds, forcing_terms, fa_outputs

    def train(self, trajectory):
        """Train a DMP with a trajectory.

        Args:
            trajectory - The trajectory with which to train the DMP.
        """
        # Set tau, initial_state and attractor_state from the trajectory
        self.tau = trajectory.ts[-1]
        self.y_init = trajectory.ys[0, :]
        self.y_attr = trajectory.ys[-1, :]

        # This needs to be computed for (optional) scaling of the forcing term.
        # Needs to be done BEFORE _compute_targets
        self._scaling_amplitudes = trajectory.get_range_per_dim()

        # Do not train function approximators if there are none
        if self._function_approximators is not None:
            (fa_input_phase, f_target) = self._compute_targets(trajectory)

            for dd in range(self.dim_dmp()):
                fa_target = f_target[:, dd]
                self._function_approximators[dd].train(fa_input_phase, fa_target)

        # Save the times steps on which the Dmp was trained.
        # This is just a convenience function to be able to call
        # analytical_solution without the "ts" argument.
        self._ts_train = trajectory.ts

    def _compute_targets(self, trajectory):
        """Given a trajectory, compute the inputs and targets for the function approximators.

        For a standard Dmp the inputs will be the phase over time, and the targets will be the
        forcing term (with the gating function factored out).

        Args:
            trajectory - Trajectory, e.g. a demonstration.

        Returns:
            fa_inputs_phase - The inputs for the function approximators (phase signal)
            fa_targets - The targets for the function approximators (forcing term)
        """

        n_time_steps = trajectory.ts.size
        dim_data = trajectory.dim
        if self.dim_dmp() != dim_data:
            raise ValueError("dims of trajectory data and dmp must be the same")

        (xs_ana, xds_ana, forcing_terms, fa_outputs) = self.analytical_solution(trajectory.ts)
        xs_goal = xs_ana[:, self.GOAL]
        xs_gating = xs_ana[:, self.GATING]
        xs_phase = xs_ana[:, self.PHASE]

        fa_inputs_phase = xs_phase

        # Get parameters from the spring-dampers system to compute inverse
        damping_coefficient = self._spring_system.damping_coefficient
        spring_constant = self._spring_system.spring_constant
        mass = self._spring_system.mass

        # Compute inverse
        tau = self._tau
        f_target = (
            tau * tau * trajectory.ydds
            + (
                spring_constant * (trajectory.ys - xs_goal)
                + damping_coefficient * tau * trajectory.yds
            )
            / mass
        )

        # Factor out gating term
        for dd in range(self.dim_dmp()):
            f_target[:, dd] = f_target[:, dd] / np.squeeze(xs_gating)

        # Factor out scaling
        if self._forcing_term_scaling == "G_MINUS_Y0_SCALING":
            g_minus_y0 = self._y_attr - self._y_init
            g_minus_y0_rep = np.tile(g_minus_y0, (n_time_steps, 1))
            f_target /= g_minus_y0_rep

        elif self._forcing_term_scaling == "AMPLITUDE_SCALING":
            _scaling_amplitudes_rep = np.tile(self._scaling_amplitudes, (n_time_steps, 1))
            f_target /= _scaling_amplitudes_rep

        return fa_inputs_phase, f_target

    def states_as_pos_vel_acc(self, x_in, xd_in):
        return x_in[self.SPRING_Y], xd_in[self.SPRING_Y], xd_in[self.SPRING_Z] / self._tau

    def states_as_trajectory(self, ts, x_in, xd_in):
        """Get the output of a DMP dynamical system as a trajectory.

        As it is a dynamical system, the state vector of a DMP contains the output of the goal,
        spring, phase and gating system. What we are most interested in is the output of the
        spring system. This function extracts that information, and also computes the
        accelerations of the spring system, which are only stored implicitly in xd_in because
        second order systems are converted to first order systems with expanded state.

        Args:
            ts    - A vector of times
            x_in  - State vector over time
            xd_in - State vector over time (rates of change)

        Return:
            Trajectory representation of the DMP state vector output.
        """
        # Left column is time
        return Trajectory(
            ts, x_in[:, self.SPRING_Y], xd_in[:, self.SPRING_Y], xd_in[:, self.SPRING_Z] / self._tau
        )

    @DynamicalSystem.tau.setter
    def tau(self, new_tau):

        self._tau = new_tau  # noqa defined inside __init__ of DynamicalSystem

        # Set value in all relevant subsystems also
        self._spring_system.tau = new_tau
        if self._goal_system is not None:
            self._goal_system.tau = new_tau
        self._phase_system.tau = new_tau
        self._gating_system.tau = new_tau

    @DynamicalSystem.y_init.setter
    def y_init(self, y_init_new):
        if y_init_new.size != self.dim_dmp():
            raise ValueError("y_init must have same size {self.dim_dmp()}")
        self._y_init = y_init_new  # noqa defined inside DynamicalSystem.__init__ of

        # Set value in all relevant subsystems also
        self._spring_system.y_init = y_init_new
        if self._goal_system is not None:
            self._goal_system.y_init = y_init_new

    @property
    def y_attr(self):
        """ Return the y part of the attractor state, where x = [y z]
        """
        return self._y_attr

    @y_attr.setter
    def y_attr(self, y_attr_new):
        if y_attr_new.size != self.dim_dmp():
            raise ValueError("y_init must have same size {self.dim_dmp()}")

        self._y_attr = y_attr_new

        # Set value in all relevant subsystems also
        if self._goal_system is not None:
            self._goal_system.y_attr = y_attr_new

        # Do NOT do the following. The attractor state of the spring system is
        # determined by the goal system.
        # self._spring_system.y_attr = y_attr_new

    @property
    def ts_train(self):
        return self._ts_train

    def set_selected_param_names(self, names):
        if isinstance(names, str):
            names = [names]  # Convert to list

        if "goal" in names:
            self._selected_param_names = ["goal"]
            # No need to bother function approximators with it: remove all occurrences
            names = [n for n in names if n != "goal"]

        # Any remaining names are passed to all function approximators
        for fa in self._function_approximators:
            fa.set_selected_param_names(names)

    def get_param_vector(self):
        values = np.empty(0)
        for fa in self._function_approximators:
            if fa.is_trained():
                values = np.append(values, fa.get_param_vector())
        if "goal" in self._selected_param_names:
            values = np.append(values, self._y_attr)
        return values

    def set_param_vector(self, values):
        size = self.get_param_vector_size()
        if len(values) != size:
            raise ValueError("values must have size {size}")
        offset = 0
        for fa in self._function_approximators:
            if fa.is_trained():
                cur_size = fa.get_param_vector_size()
                cur_values = values[offset : offset + cur_size]
                fa.set_param_vector(cur_values)
                offset += cur_size
        if "goal" in self._selected_param_names:
            self.y_attr = values[offset : offset + self.dim_dmp()]

    def get_param_vector_size(self):
        size = 0
        for fa in self._function_approximators:
            if fa.is_trained():
                size += fa.get_param_vector_size()
        if "goal" in self._selected_param_names:
            size += self.dim_dmp()
        return size

    @staticmethod
    def get_dmp_axes(has_fa_output=False):
        n_cols = 5
        n_rows = 3 if has_fa_output else 2
        fig = plt.figure(figsize=(3 * n_cols, 3 * n_rows))

        axs = [fig.add_subplot(n_rows, 5, i + 1) for i in range(n_rows * 5)]
        return axs

    def plot(self, ts, xs, xds, **kwargs):
        forcing_terms = kwargs.get("forcing_terms", [])
        fa_outputs = kwargs.get("fa_outputs", [])
        ext_dims = kwargs.get("ext_dims", [])
        plot_tau = kwargs.get("plot_tau", True)
        has_fa_output = len(forcing_terms) > 0 or len(fa_outputs) > 0

        axs = kwargs.get("axs") or Dmp.get_dmp_axes(has_fa_output)

        d = self.dim_dmp()  # noqa Abbreviation for convenience
        systems = [
            ("phase", range(3 * d, 3 * d + 1), axs[0:2], self._phase_system),
            ("gating", range(3 * d + 1, 3 * d + 2), axs[5:7], self._gating_system),
            ("goal", range(2 * d, 3 * d), axs[2:4], self._gating_system),
            ("spring-damper", range(0 * d, 2 * d), axs[7:10], self._spring_system),
        ]
        # system_varname = ["x", "v", "y^{g_d}", "y"]

        all_handles = []
        for system in systems:
            xs_cur = xs[:, system[1]]
            xds_cur = xds[:, system[1]]
            axs_cur = system[2]
            h, _ = system[3].plot(ts, xs_cur, xds_cur, axs=axs_cur)
            all_handles.append(h)
            for i, ax in enumerate(axs_cur):
                x = np.mean(ax.get_xlim())
                y = np.mean(ax.get_ylim())
                ax.text(x, y, system[0], horizontalalignment="center")
                if plot_tau:
                    ax.plot([self._tau, self._tau], ax.get_ylim(), "--k")

        if len(fa_outputs) > 1:
            ax = axs[11 - 1]
            h = ax.plot(ts, fa_outputs)
            all_handles.extend(h)
            x = np.mean(ax.get_xlim())
            y = np.mean(ax.get_ylim())
            ax.text(x, y, "func. approx.", horizontalalignment="center")
            ax.set_xlabel(r"time ($s$)")
            ax.set_ylabel(r"$f_\mathbf{\theta}(x)$")

        if len(forcing_terms) > 1:
            ax = axs[12 - 1]
            h = ax.plot(ts, forcing_terms)
            all_handles.extend(h)
            x = np.mean(ax.get_xlim())
            y = np.mean(ax.get_ylim())
            ax.text(x, y, "forcing term", horizontalalignment="center")
            ax.set_xlabel(r"time ($s$)")
            ax.set_ylabel(r"$v\cdot f_{\mathbf{\theta}}(x)$")

        if len(ext_dims) > 1:
            ax = axs[13 - 1]
            h = ax.plot(ts, ext_dims)
            all_handles.extend(h)
            x = np.mean(ax.get_xlim())
            y = np.mean(ax.get_ylim())
            ax.text(x, y, "extended dims", horizontalalignment="center")
            ax.set_xlabel(r"time ($s$)")
            ax.set_ylabel(r"unknown")

        return all_handles, axs
