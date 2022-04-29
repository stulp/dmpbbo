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

import copy

import numpy as np

from dmpbbo.bbo_for_dmps.TaskSolver import TaskSolver


class TaskSolverDmp(TaskSolver):
    def __init__(self, dmp, dt, integrate_dmp_beyond_tau_factor):
        self._dmp = copy.deepcopy(dmp)
        self._integrate_time = dmp.tau * integrate_dmp_beyond_tau_factor
        self._n_time_steps = int(np.floor(self._integrate_time / dt)) + 1

    def perform_rollout_dmp(self, dmp):
        ts = np.linspace(0.0, self._integrate_time, self._n_time_steps)
        xs, xds, forcing_terms, fa_outputs = dmp.analytical_solution(ts)
        traj = dmp.states_as_trajectory(ts, xs, xds)
        # traj.misc = forcing_terms
        cost_vars = traj.as_matrix()
        return cost_vars

    def perform_rollout(self, sample, **kwargs):
        self._dmp.set_param_vector(sample)
        return self.perform_rollout_dmp(self._dmp)
