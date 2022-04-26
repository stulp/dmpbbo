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
        self.dmp_ = copy.deepcopy(dmp)
        self.integrate_time_ = dmp.tau * integrate_dmp_beyond_tau_factor
        self.n_time_steps_ = int(np.floor(self.integrate_time_ / dt)) + 1

    def performRolloutDmp(self, dmp, **kwargs):
        ts = np.linspace(0.0, self.integrate_time_, self.n_time_steps_)
        xs, xds, forcing_terms, fa_outputs = dmp.analyticalSolution(ts)
        traj = dmp.statesAsTrajectory(ts, xs, xds)
        # traj.misc = forcing_terms
        cost_vars = traj.asMatrix()
        return cost_vars

    def performRollout(self, sample, **kwargs):
        self.dmp_.setParamVector(sample)
        return self.performRolloutDmp(self.dmp_, **kwargs)
