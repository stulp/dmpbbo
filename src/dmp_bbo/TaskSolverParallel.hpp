/**
 * @file   TaskSolverParallel.hpp
 * @brief  TaskSolverParallel class header file.
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
 
#ifndef TASK_SOLVER_PARALLEL_H
#define TASK_SOLVER_PARALLEL_H
#define EIGEN2_SUPPORT
#include "bbo/TaskSolver.hpp"

#include <vector>
#include <eigen3/Eigen/Core>

#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

namespace DmpBbo {

/** Interface for classes that can perform rollouts.
 * For further information see the section on \ref sec_bbo_task_and_task_solver
 */
 class TaskSolverParallel : public TaskSolver
{
public:
    
  /** Perform rollouts, i.e. given a set of samples, determine all the variables that are relevant to evaluating the cost function. 
   * \param[in] samples The samples
   * \param[in] task_parameters The parameters of the task
   * \param[out] cost_vars The variables relevant to computing the cost.
   * \todo Compare to other functions
   */
  inline void performRollouts(const Eigen::MatrixXd& samples, const Eigen::MatrixXd& task_parameters, Eigen::MatrixXd& cost_vars) const 
  {
    std::vector<Eigen::MatrixXd> samples_vec(1);
    samples_vec[0] = samples;
    performRollouts(samples_vec, cost_vars);
  };
  
  /** Perform rollouts, i.e. given a set of samples, determine all the variables that are relevant
   * to evaluating the cost function. 
   * This version does so for parallel optimization, where multiple distributions are updated.
   * \param[in] samples_vec The samples, a vector with one element per distribution
   * \param[out] cost_vars The variables relevant to computing the cost.
   * \todo Compare to other functions
   */
  void performRollouts(const std::vector<Eigen::MatrixXd>& samples_vec, Eigen::MatrixXd& cost_vars) const 
  {
    Eigen::MatrixXd task_parameters(0,0);
    performRollouts(samples_vec,task_parameters,cost_vars);
  }
  
  /** Perform rollouts, i.e. given a set of samples, determine all the variables that are relevant
   * to evaluating the cost function. 
   * This version does so for parallel optimization, where multiple distributions are updated.
   * \param[in] samples_vec The samples, a vector with one element per distribution
   * \param[in] task_parameters The parameters of the task
   * \param[out] cost_vars The variables relevant to computing the cost.
   * \todo Compare to other functions
   */
  virtual void performRollouts(const std::vector<Eigen::MatrixXd>& samples_vec, const Eigen::MatrixXd& task_parameters, Eigen::MatrixXd& cost_vars) const = 0;
  
  /** Print a TaskSolver to an output stream. 
   *
   *  \param[in] output  Output stream to which to write to
   *  \param[in] task_solver TaskSolver to write
   *  \return    Output stream
   *
   *  \remark Calls virtual function TaskSolver::toString, which must be implemented by
   * subclasses: http://stackoverflow.com/questions/4571611/virtual-operator
   */ 
  friend std::ostream& operator<<(std::ostream& output, const TaskSolverParallel& task_solver) {
    output << task_solver.toString();
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
    // serialize base class information
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(TaskSolver);
  }
  
};
  
} // namespace DmpBbo

#include <boost/serialization/assume_abstract.hpp>
/** Don't add version information to archives. */
BOOST_SERIALIZATION_ASSUME_ABSTRACT(DmpBbo::TaskSolverParallel);
 
#include <boost/serialization/export.hpp>
/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::TaskSolverParallel,boost::serialization::object_serializable);

#endif

