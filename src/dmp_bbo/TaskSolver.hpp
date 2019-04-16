/**
 * @file   TaskSolver.hpp
 * @brief  TaskSolver class header file.
 * @author Freek Stulp
 *
 * This file is part of DmpBbo, a set of libraries and programs for the 
 * black-box optimization of dynamical movement primitives.
 * Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
 * 
 * DmpBbo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 * 
 * DmpBbo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
 */
 
#ifndef TASK_SOLVER_H
#define TASK_SOLVER_H

#include <eigen3/Eigen/Core>

namespace DmpBbo {

/** Interface for classes that can perform rollouts.
 * For further information see the section on \ref sec_bbo_task_and_task_solver
 */
class TaskSolver
{
public:
  
  /** Perform a rollout, i.e. given a sample, determine all the variables that are relevant to evaluating the cost function. 
   * See also \ref sec_cost_vars and \ref sec_bbo_task_and_task_solver
   * \param[in] sample The samples
   * \param[out] cost_vars The variables relevant to computing the cost.
   */
  void performRollout(const Eigen::VectorXd& sample, Eigen::MatrixXd& cost_vars) const;
    
  /** Perform a rollout, i.e. given a sample, determine all the variables that are relevant to evaluating the cost function. 
   * See also \ref sec_cost_vars and \ref sec_bbo_task_and_task_solver
   * \param[in] sample The samples
   * \param[in] task_parameters The parameters of the task
   * \param[out] cost_vars The variables relevant to computing the cost.
   */
  virtual void performRollout(const Eigen::VectorXd& sample, const Eigen::VectorXd& task_parameters, Eigen::MatrixXd& cost_vars) const = 0;
  
  /** Returns a string representation of the object.
   * \return A string representation of the object.
   */
  virtual std::string toString(void) const = 0;

  /** Print a TaskSolver to an output stream. 
   *
   *  \param[in] output  Output stream to which to write to
   *  \param[in] task_solver TaskSolver to write
   *  \return    Output stream
   *
   *  \remark Calls virtual function TaskSolver::toString, which must be implemented by
   * subclasses: http://stackoverflow.com/questions/4571611/virtual-operator
   */ 
  friend std::ostream& operator<<(std::ostream& output, const TaskSolver& task_solver) {
    output << task_solver.toString();
    return output;
  }
  
};

} // namespace DmpBbo

#endif

