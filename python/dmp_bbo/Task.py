# This file is part of DmpBbo, a set of libraries and programs for the
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
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

from abc import ABC, abstractmethod


class Task(ABC):
    """Interface for cost functions, which define a task.
    For further information see the section on \ref sec_bbo_task_and_task_solver
    """

    @abstractmethod
    def evaluateRollout(self, cost_vars, sample):
        """The cost function which defines the task.
       
        \param[in] cost_vars All the variables relevant to computing the cost. These are determined by TaskSolver::performRollout(). For further information see the section on \ref sec_bbo_task_and_task_solver
        \param[in] sample The sample from which the rollout was generated. Passing this to the cost function is useful when performing regularization on the sample. For further information see the section on \ref sec_bbo_task_and_task_solver
         \return costs The scalar cost components for the sample. The first item costs[0] should contain the total cost.
        """
        pass

    def plotRollout(self, cost_vars, ax=None):
        if not ax:
            ax = plt.axes()
        h = plot(cost_vars, "-")
        return (h, ax)

    def costLabels(self):
        """Labels for the different cost components.
        
        The cost function evaluateRollout may return an array of costs. The first one cost[0] is always the sum of the other ones, i.e. costs[0] = sum(costs[1:]). This function returns labels for the individual cost components.
        """
        return None
