/**
 * @file   Task.hpp
 * @brief  Task class header file.
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
 
#ifndef TASK_H
#define TASK_H

#include <vector>
#include <eigen3/Eigen/Core>

#include <boost/serialization/access.hpp>

namespace DmpBbo {

/** Interface for cost functions, which define a task.
 * For further information see the section on \ref sec_bbo_task_and_task_solver
 */
class Task
{
public:
  /** The cost function which defines the task.
   * See also \ref sec_cost_components and \ref sec_bbo_task_and_task_solver
   *
   * \param[in] cost_vars All the variables relevant to computing the cost. These are determined by TaskSolver::performRollout(). For further information see the section on \ref sec_bbo_task_and_task_solver
   * \param[in] sample The sample from which cost_vars was generated. Required for regularization.
   * \param[out] costs The cost for these cost_vars. The first element cost[0] should be the total cost. The others may be the individual cost components that consitute the total cost, e.g. cost[0] = cost[1] + cost[2] ...
   */
  virtual void evaluateRollout(const Eigen::MatrixXd& cost_vars, const Eigen::VectorXd& sample, Eigen::VectorXd& costs) const 
  {
    int n_task_pars = 0;
    Eigen::VectorXd task_parameters(n_task_pars);
    evaluateRollout(cost_vars,sample,task_parameters,costs);
  };
  
  virtual unsigned int getNumberOfCostComponents(void) const = 0;
  
  /** The cost function which defines the task.
   * See also \ref sec_cost_components and \ref sec_bbo_task_and_task_solver
   * \param[in] cost_vars All the variables relevant to computing the cost. These are determined by TaskSolver::performRollout(). For further information see the section on \ref sec_bbo_task_and_task_solver
   * \param[in] sample The sample from which cost_vars was generated. Required for regularization.
   * \param[in] task_parameters Optional parameters of the task, and thus the cost function.
   * \param[out] cost The cost for these cost_vars. The first element should be the total cost. The others may be different cost components.
   */
  virtual void evaluateRollout(const Eigen::MatrixXd& cost_vars, const Eigen::VectorXd& sample, const Eigen::VectorXd& task_parameters, Eigen::VectorXd& cost) const = 0;
  
  /** Save a python script that is able to visualize the rollouts, given the cost-relevant variables
   *  stored in a file.
   *  \param[in] directory Directory in which to save the python script
   *  \return true if saving the script was successful, false otherwise
   */
  virtual bool savePlotRolloutScript(std::string directory) const
  {
    return true;
  }
  
  /** Returns a string representation of the object.
   * \return A string representation of the object.
   */
  virtual std::string toString(void) const = 0;

  /** Write to output stream. 
   *  \param[in] output Output stream to which to write to 
   *  \param[in] task Task object to write
   *  \return Output to which the object was written 
   *
   *  \remark Calls virtual function Task::toString, which must be implemented by
   * subclasses: http://stackoverflow.com/questions/4571611/virtual-operator
   */ 
  friend std::ostream& operator<<(std::ostream& output, const Task& task) {
    output << task.toString();
    return output;
  }
  
private:
  /** Give boost serialization access to private members. */  
  friend class boost::serialization::access;
  
  /** Serialize class data members to boost archive. 
   * \param[in] ar Boost archive
   * \param[in] version Version of the class
   * See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/tutorial.html#simplecase
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    // No members to serialize.
  }
  
};

} // namespace DmpBbo

#include <boost/serialization/assume_abstract.hpp>
/** Don't add version information to archives. */
BOOST_SERIALIZATION_ASSUME_ABSTRACT(DmpBbo::Task);
 
#include <boost/serialization/level.hpp>
/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::Task,boost::serialization::object_serializable);

#endif

