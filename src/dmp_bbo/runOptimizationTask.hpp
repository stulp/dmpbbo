/**
 * @file runOptimizationTask.hpp
 * @brief  Header file for function to run an evolutionary optimization process with a task.
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

#ifndef RUNEVOLUTIONARYOPTIMIZATIONTASK_H
#define RUNEVOLUTIONARYOPTIMIZATIONTASK_H

#include <string>
#include <vector>
#include <eigen3/Eigen/Core>


namespace DmpBbo {
  
// Forward declarations
class DistributionGaussian;
class Updater;
class Task;
class TaskSolver;
class ExperimentBBO;
class Rollout;

/** Run an evolutionary optimization process, see \ref page_bbo
 * \param[in] task The Task to optimize
 * \param[in] task_solver The TaskSolver that will solve the task
 * \param[in] initial_distribution The initial parameter distribution
 * \param[in] updater The Updater used to update the parameters
 * \param[in] n_updates The number of updates to perform
 * \param[in] n_samples_per_update The number of samples per update
 * \param[in] save_directory Optional directory to save to (default: don't save)
 * \param[in] overwrite Overwrite existing files in the directory above (default: false)
 * \param[in] only_learning_curve Save only the learning curve (default: false)
 */
void runOptimizationTask(
  const Task* const task, 
  const TaskSolver* const task_solver, 
  const DistributionGaussian* const initial_distribution, 
  const Updater* const updater, 
  int n_updates, 
  int n_samples_per_update, 
  std::string save_directory=std::string(""),
  bool overwrite=false,
  bool only_learning_curve=false);

/** Run an evolutionary optimization process, see \ref page_bbo
 * \param[in] experiment The experiment to run, cf. ExperimentBBO
 * \param[in] save_directory Optional directory to save to (default: don't save)
 * \param[in] overwrite Overwrite existing files in the directory above (default: false)
 * \param[in] only_learning_curve Save only the learning curve (default: false)
 */
void runOptimizationTask(
  ExperimentBBO* experiment, 
  std::string save_directory=std::string(""),
  bool overwrite=false,
  bool only_learning_curve=false);

/** Run several parallel evolutionary optimization processes.
 * \param[in] task The task to optimize
 * \param[in] task_solver The solver of the task
 * \param[in] initial_distributions The initial parameter distribution (one for each parallel optimization)
 * \param[in] updater The Updater used to update the parameters
 * \param[in] n_updates The number of updates to perform
 * \param[in] n_samples_per_update The number of samples per update
 * \param[in] save_directory Optional directory to save to (default: don't save)
 * \param[in] overwrite Overwrite existing files in the directory above (default: false)
 * \param[in] only_learning_curve Save only the learning curve (default: false)
 * \todo This function will be removed. It can be implemented much simpler. Only UpdateCovarAdapation is affected by this, and one can enforce block-diagonal-covars there (e.g. with a function setSubCovars(VectorXi). Update the docu accordingly when this is done. )
 */
void runOptimizationParallelDeprecated(Task* task, TaskSolver* task_solver, std::vector<DistributionGaussian*> initial_distributions, Updater* updater, int n_updates, int n_samples_per_update, std::string save_directory=std::string(""),bool overwrite=false,
bool only_learning_curve=false);

/** Save the results of one update to a directory
 * \param[in] directory Directory to save to
 * \param[in] i_update Current update number. E.g. if i_update=9, results are stored to directory/update00009/
 * \param[in] distribution Gaussian distribution before updating
 * \param[in] rollout_eval Evaluation rollout, i.e. without exploration noise
 * \param[in] rollouts Exploration rollouts in this update
 * \param[in] weights Weights resulting from the costs of the rollouts
 * \param[in] distribution_new Gaussian distribution after updating
 * \param[in] overwrite Overwrite existing files in the directory above (default: false)
 * \return true is saving was successful, false otherwise.
 */
bool saveToDirectory(std::string directory, int i_update, const DistributionGaussian& distribution, const Rollout* rollout_eval, const std::vector<Rollout*>& rollouts, const Eigen::VectorXd& weights, const DistributionGaussian& distribution_new, bool overwrite=false);

/** Save the results of one update to a directory
 * \param[in] directory Directory to save to
 * \param[in] i_update Current update number. E.g. if i_update=9, results are stored to directory/update00009/
 * \param[in] distributions Gaussian distributions before updating
 * \param[in] rollout_eval Evaluation rollout, i.e. without exploration noise
 * \param[in] rollouts Exploration rollouts in this update
 * \param[in] weights Weights resulting from the costs of the rollouts
 * \param[in] distributions_new Gaussian distributions after updating
 * \param[in] overwrite Overwrite existing files in the directory above (default: false)
 * \return true is saving was successful, false otherwise.
 */
bool saveToDirectory(std::string directory, int i_update, const std::vector<DistributionGaussian>& distributions, const Rollout* rollout_eval, const std::vector<Rollout*>& rollouts, const Eigen::VectorXd& weights, const std::vector<DistributionGaussian>& distributions_new, bool overwrite=false);

}

#endif

/** \defgroup DMP_BBO Black Box Optimization of Dynamical Movement Primitives Module
 */

 /**
\verbatim
For standard optimization, n_parallel = 1 and n_time_steps = 1 so that
                    vector<Matrix>            Matrix
  samples         =                   n_samples x n_dim
  task_parameters =                   n_samples x n_task_pars
  cost_vars       =                   n_samples x n_cost_vars


Generic case: n_dofs-D Dmp, n_parallel=n_dofs
                    vector<Matrix>            Matrix
  samples         = n_parallel     x  n_samples x sum(n_model_parameters)
  task_parameters =                   n_samples x n_task_pars
  cost_vars       =                   n_samples x (n_time_steps*n_cost_vars)
  
Standard optimization is special case of the above with, n_parallel = 1 and n_time_steps = 1
\endverbatim
 
 */
 

