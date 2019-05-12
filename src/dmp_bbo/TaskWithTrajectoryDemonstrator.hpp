/**
 * @file   TaskWithTrajectoryDemonstrator.hpp
 * @brief  TaskWithTrajectoryDemonstrator class header file.
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

#ifndef TASKWITHTRAJECTORYDEMONSTRATOR_H
#define TASKWITHTRAJECTORYDEMONSTRATOR_H

#include "dmp_bbo/Task.hpp"
#include "bbo/DistributionGaussian.hpp"
#include "dmp/Trajectory.hpp"

#include <vector>
#include <eigen3/Eigen/Core>

namespace DmpBbo {
  
/** Interface for tasks that are able to provide demonstrations that solve the task (optimally).
 * Such tasks must implement the pure virtual function TaskWithTrajectoryDemonstrator::generateDemonstration(), which takes task_parmetersm and returns a demonstration. 
 */
class TaskWithTrajectoryDemonstrator : public Task
{
public:
  
  /** Generate one (optimal) demonstration for this task.
   * \param[in] task_parameters The task parameters for which to generate a demonstration. A matrix of size T X D, where T is the number of time steps, and D is the number of task parameters. If T=1, the task parameters are assumed to be constant over time.
   * \param[in] ts The times at which to sample the trajectory
   * \param[out] demonstration The demonstration trajectory
   */
  virtual void generateDemonstration(const Eigen::MatrixXd& task_parameters, const Eigen::VectorXd& ts, Trajectory& demonstration) const = 0;
  
  /** Generate a set of demonstrations for this task.
   * \param[in] task_parameters The task parameters for which to generate demonstrations
   * \param[in] ts The times at which to sample the trajectory
   * \param[out] demonstrations The demonstrations (a vector of trajectories)
   */
  void generateDemonstrations(const std::vector<Eigen::MatrixXd>& task_parameters, const std::vector<Eigen::VectorXd>& ts, std::vector<Trajectory>& demonstrations) const;
  
  /** Generate a set of demonstrations for this task.
   * \param[in] task_parameter_distribution The distribution from which to sample task parameters
   * \param[in] n_demos The number of demos
   * \param[in] ts The times at which to sample the trajectory
   * \param[out] demonstrations The demonstrations (a vector of trajectories)
   */
  void generateDemonstrations(DistributionGaussian* task_parameter_distribution, int n_demos, const Eigen::VectorXd& ts, std::vector<Trajectory>& demonstrations) const;

  /*
  Matlab code:
  void generatedemonstrationsrandom(int n_demonstrations, Matrix&task_instances)
  {
     // Generate random task parameters
     task_parameters = obj.task_parameters_distribution.getsamples(n_demonstrations);
     // Generate task instances
     generatedemonstrations(task_parameters,task_instances);
  }
    
    function task_instances = generatedemonstrationsgrid(obj,n_demonstrations)
      n_dim = obj.task_parameters_distribution.n_dims;

      % Process arguments
      if (length(n_demonstrations)==1)
         % Copy value to make array of length n_dim
        n_demonstrations = n_demonstrations*ones(1,n_dim);
      elseif (length(n_demonstrations)==n_dim)
        % Everything is fine
      else
        error('n_demonstrations should be of length 1 or n_dim') %#ok<WNTAG>
      end

      % Generate task parameters on a grid
      task_parameters = obj.task_parameters_distribution.getgridsamples(n_demonstrations);
      
      % Remove duplicate rows (might be due to redundant dimensions)
      task_parameters = unique(task_parameters,'rows');

      % Generate task instances
      task_instances = obj.generatedemonstrations(task_parameters);
   
    end
  end
  */
  
  
};
  
} // namespace DmpBbo

#endif

