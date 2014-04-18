/**
 * @file   ExperimentBBO.hpp
 * @brief  ExperimentBBO class header file.
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
#define EIGEN2_SUPPORT
#ifndef EXPERIMENTBBO_H
#define EXPERIMENTBBO_H

#include <vector>
#include <eigen3/Eigen/Core>

#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>

namespace DmpBbo {

class CostFunction;
class Task;
class TaskSolver;
class DistributionGaussian;
class Updater;
  
/** POD class to store all objects relevant to running and evolutionary optimization. */
class ExperimentBBO
{

public:
  /** Constructor.
   * \param[in] cost_function_arg The cost function to optimize
   * \param[in] initial_distribution_arg The initial parameter distribution
   * \param[in] updater_arg The Updater used to update the parameters
   * \param[in] n_updates_arg The number of updates to perform
   * \param[in] n_samples_per_update_arg The number of samples per update
   */
  ExperimentBBO(
    CostFunction* cost_function_arg,
    DistributionGaussian* initial_distribution_arg,
    Updater* updater_arg,
    int n_updates_arg,
    int n_samples_per_update_arg
  )
  :
    cost_function(cost_function_arg),
    task(NULL),
    task_solver(NULL),
    initial_distribution(initial_distribution_arg),
    updater(updater_arg),
    n_updates(n_updates_arg),
    n_samples_per_update(n_samples_per_update_arg)
  {}
  
  /** Constructor.
   * \param[in] task_arg The Task to optimize
   * \param[in] task_solver_arg The TaskSolver that will solve the task
   * \param[in] initial_distribution_arg The initial parameter distribution
   * \param[in] updater_arg The Updater used to update the parameters
   * \param[in] n_updates_arg The number of updates to perform
   * \param[in] n_samples_per_update_arg The number of samples per update
   */
  ExperimentBBO(
    Task* task_arg,
    TaskSolver* task_solver_arg,
    DistributionGaussian* initial_distribution_arg,
    Updater* updater_arg,
    int n_updates_arg,
    int n_samples_per_update_arg
  )
  :
    cost_function(NULL),
    task(task_arg),
    task_solver(task_solver_arg),
    initial_distribution(initial_distribution_arg),
    updater(updater_arg),
    n_updates(n_updates_arg),
    n_samples_per_update(n_samples_per_update_arg)
  {}

  /** Cost function to be used during evaluation.
   * If it is NULL, ExperimentBBO::task and ExperimentBBO::task_solver must be set.
   */
  const CostFunction* cost_function;
  
  /** Task to be used during evaluation.
   * If it is NULL, ExperimentBBO::cost_function must be set.
   */
  const Task* task;

  /** Task solver to be used for a rollout.
   * If it is NULL, ExperimentBBO::cost_function must be set.
   */
  const TaskSolver* task_solver;
  
  /** The initial parameter distribution for the search. */
  const DistributionGaussian* const initial_distribution; 

  /** The updater used to update the parameters of the distribution. */
  const Updater* updater;

  /** The number of updates to perform. */
  int n_updates;
  
  /** The number of samples per update. */
  int n_samples_per_update;

  /** Serialize class data members to boost archive. 
   * \param[in] ar Boost archive
   * \param[in] version Version of the class
   * See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/tutorial.html#simplecase
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & BOOST_SERIALIZATION_NVP(cost_function);
    ar & BOOST_SERIALIZATION_NVP(task);
    ar & BOOST_SERIALIZATION_NVP(task_solver);
    ar & BOOST_SERIALIZATION_NVP(initial_distribution);
    ar & BOOST_SERIALIZATION_NVP(updater);
    ar & BOOST_SERIALIZATION_NVP(n_updates);
    ar & BOOST_SERIALIZATION_NVP(n_samples_per_update);
  }

};

}

#include <boost/serialization/level.hpp>
/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::ExperimentBBO,boost::serialization::object_serializable);

/*
class ExperimentBBOResults
{
  ExperimentBBO experiment;
  vector<UpdateSummary> update_summaries;
  vector<vector<MatrixXd>> cost_vars;
}
*/


#endif 
