/**
 * @file runEvolutionaryOptimization.hpp
 * @brief  Header file for function to run an evolutionary optimization process.
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

#ifndef RUNEVOLUTIONARYOPTIMIZATION_H
#define RUNEVOLUTIONARYOPTIMIZATION_H

#include <string>
#include <vector>


namespace DmpBbo {
  
// Forward declarations
class DistributionGaussian;
class Updater;
class CostFunction;
class Task;
class TaskSolver;
class ExperimentBBO;

/** Run an evolutionary optimization process, see \ref page_bbo
 * \param[in] cost_function The cost function to optimize
 * \param[in] initial_distribution The initial parameter distribution
 * \param[in] updater The Updater used to update the parameters
 * \param[in] n_updates The number of updates to perform
 * \param[in] n_samples_per_update The number of samples per update
 * \param[in] save_directory Optional directory to save to (default: don't save)
 * \param[in] overwrite Overwrite existing files in the directory above (default: false)
 */
void runEvolutionaryOptimization(
  const CostFunction* const cost_function, 
  const DistributionGaussian* const initial_distribution, 
  const Updater* const updater, 
  int n_updates, 
  int n_samples_per_update, 
  std::string save_directory=std::string(""),
  bool overwrite=false,bool only_learning_curve=false);

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
void runEvolutionaryOptimization(
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
void runEvolutionaryOptimization(
  ExperimentBBO* experiment, 
  std::string save_directory=std::string(""),
  bool overwrite=false,
  bool only_learning_curve=false);

}

#endif

/** \defgroup BBO Black Box Optimization Module
 */

/** \page page_bbo Black Box Optimization

This module implements several <A HREF="http://en.wikipedia.org/wiki/Evolution_strategy">evolution strategies</A> for the <A HREF="http://en.wikipedia.org/wiki/Optimization_%28mathematics%29">optimization</A> of black-box <A HREF="http://en.wikipedia.org/wiki/Loss_function">cost functions</A>. Black-box in this context means that no assumptions about the cost function can be made, for example, we do not have access to its derivative, and we do not even know if it is continuous or not.

The evolution strategies that are implemented are all based on reward-weighted averaging (aka probablity-weighted averaging), as explained in this paper/presentation: http://icml.cc/discuss/2012/171.html

The basic algorithm is as follows:
\code
x_mu = ??; x_Sigma = ?? // Initialize multi-variate Gaussian distribution
while (!halt_condition) {

    // Explore
    for k=1:K {
        x[k]     ~ N(x_mu,x_Sigma)    // Sample from Gaussian
        costs[k] = costfunction(x[k]) // Evaluate sample
    }
        
    // Update distribution
    weights = costs2weights(costs) // Should assign higher weights to lower costs
    x_mu_new = weights^T * x; // Compute weighted mean of samples
    x_covar_new = (weights .* x)^T * weights // Compute weighted covariance matrix of samples
    
    x_mu = x_mu_new
    x_covar = x_covar_new
}
\endcode

\section sec_bbo_implementation Implementation

The algorithm above has been implemented as follows (see 
runEvolutionaryOptimization() and demoEvolutionaryOptimization.cpp):
\code

int n_dim = 2; // Optimize 2D problem

// This is the cost function to be optimized
CostFunction* cost_function = new CostFunctionQuadratic(VectorXd::Zero(n_dim));

// This is the initial distribution
DistributionGaussian* distribution = new DistributionGaussian(VectorXd::Random(n_dim),MatrixXd::Identity(n_dim)) 

// This is the updater which will update the distribution
double eliteness = 10.0;
Updater* updater = new UpdaterMean(eliteness);

// Some variables
MatrixXd samples;
VectorXd costs;

for (int i_update=1; i_update<=n_updates; i_update++)
{
  
    // 1. Sample from distribution
    int n_samples_per_update = 10;
    distribution->generateSamples(n_samples_per_update, samples);
  
    // 2. Evaluate the samples
    cost_function->evaluate(samples,costs);
  
    // 3. Update parameters
    updater->updateDistribution(*distribution, samples, costs, *distribution);
    
}
\endcode

\subsection sec_bbo_task_and_task_solver CostFunction vs Task/TaskSolver

When the cost function has a simple structure, e.g. cost = \f$ x^2 \f$ it is convenient to implement the function \f$ x^2 \f$ in CostFunction::evaluate(). In robotics however, it is more suitable to make the distinction between a task (e.g. lift an object), and an entity that solves this task (e.g. your robot, my robot, a simulated robot, etc.). For these cases, the CostFunction is split into a Task and a TaskSolver, as follows:

\code
CostFunction::evaluate(samples,costs) {
  TaskSolver::performRollout(samples,cost_vars)
  Task::evaluate(cost_vars,costs)
}
\endcode

For more details, see \ref sec_dmp_bbo_task_and_task_solver

 */

