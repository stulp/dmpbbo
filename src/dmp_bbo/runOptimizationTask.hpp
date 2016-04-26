/**
 * @file runOptimization.hpp
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
 * \param[in] distributions The initial parameter distribution (one for each parallel optimization)
 * \param[in] updater The Updater used to update the parameters
 * \param[in] n_updates The number of updates to perform
 * \param[in] n_samples_per_update The number of samples per update
 * \param[in] save_directory Optional directory to save to (default: don't save)
 * \param[in] overwrite Overwrite existing files in the directory above (default: false)
 * \param[in] only_learning_curve Save only the learning curve (default: false)
 */
void runOptimizationParallel(Task* task, TaskSolver* task_solver, std::vector<DistributionGaussian*> distributions, Updater* updater, int n_updates, int n_samples_per_update, std::string save_directory=std::string(""),bool overwrite=false,
bool only_learning_curve=false);

bool saveToDirectory(std::string directory, int i_update, const DistributionGaussian& distribution, const Rollout* rollout_eval, const std::vector<Rollout*>& rollouts, const Eigen::VectorXd& weights, const DistributionGaussian& distribution_new, bool overwrite=false);

bool saveToDirectory(std::string directory, int i_update, const std::vector<DistributionGaussian>& distribution, const Rollout* rollout_eval, const std::vector<Rollout*>& rollouts, const Eigen::VectorXd& weights, const std::vector<DistributionGaussian>& distribution_new, bool overwrite=false);

}

#endif

// ZZ Merge docu below
/** \defgroup BBO Black Box Optimization Module
 */


/** \defgroup DMP_BBO Black Box Optimization of Dynamical Movement Primitives Module
 */

/** \page page_dmp_bbo Black Box Optimization of Dynamical Movement Primitives

This page assumes you have read \ref sec_bbo_implementation . Applying BBO to robots (which may execute DMPs) requires several extensions:

\li Use of a Task/TaskSolver instead of a CostFunction
\li Running multiple optimizations in parallel, one for each DOF of the DMP
\li Allowing the optimization to be run one update at a time, instead of one loop

\section sec_dmp_bbo_task_and_task_solver Task/TaskSolver

See also \ref sec_bbo_task_and_task_solver

In robotics, it is suitable to make the distinction between a task (e.g. lift an object), and an entity that solves this task (e.g. your robot, my robot, a simulated robot, etc.). For these cases, the CostFunction is split into a Task and a TaskSolver, as follows:

\code
CostFunction::evaluate(samples,costs) {
  TaskSolver::performRollouts(samples,cost_vars)
  Task::evaluate(cost_vars,costs)
}
\endcode

The roles of the Task/TaskSolver are:

\li TaskSolver: performing the rollouts on the robot (based on the samples in policy space), and saving all the variables related to computing the costs into a file (called cost_vars.txt)

\li Task: determining the costs from the cost-relevant variables


The idea here is that the TaskSolver uses the samples to perform a rollout (e.g. the samples represent the parameters of a policy which is executed) and computes all the variables that are relevant to determining the cost (e.g. it records the forces at the robot's end-effector, if this is something that needs to be minimized)

Some further advantages of this approach, which are discussed in more detail in \ref page_dmp_bbo :
\li Different robots can solve the exact same Task implementation of the same task.
\li Robots do not need to know about the cost function to perform rollouts (and they shouldn't)
\li The intermediate cost-relevant variables can be stored to file for visualization etc.
\li The procedures for performing the roll-outs (on-line on a robot) and doing the evaluation/updating/sampling (off-line on a computer) can be seperated, because there is a separate TaskSolver::performRollouts function.

When using the Task/TaskSolver approach, the runOptimization process is as follows (only minor changes to the above):
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
  
    // 2A. Perform the roll-outs
    task_solver->performRollouts(samples,cost_vars);
  
    // 2B. Evaluate the samples
    task->evaluate(cost_vars,costs);
  
    // 3. Update parameters
    updater->updateDistribution(*distribution, samples, costs, *distribution);
    
}
\endcode

\section sec_parallel_optimization Optimizing DMP DOFs in parallel

Consider a 7-DOF robotic arm. With a DMP, each DOF would be represented by a different dimension of a 7-D DMP (let as assume that each dimension of the DMP uses 10 basis functions). There are then two distinct approaches towards optimizing the parameters of the DMP with respect to a cost function:

\li Consider the search space for optimization to be 70-D, i.e. consider all open parameters of the DMP in one search space. The disadvantage is that reward-weighted averaging will require more samples to perform a robust update, especially for the covariance matrix, which is 70X70! With this approach the implementation in runOptimization and UpdateSummary is appropriate.

\li Run a seperate optimization for each of the 7 DOFs. Thus, here we have seven 10-D search spaces (with different distributions and samples for each DOF), but using the same costs for all optimizations. This is what is proposed for instance in PI^2. This approach is not able to exploit covariance between different DOFs, but in practice it works robustly.

To enable the latter approach, the function runOptimizationParallel and the class UpdateSummaryParallel have been added. The main difference is that instead of using one distribution (mean and covariance), D distributions must be used (these are stored in a std::vector).

\section sec_bbo_one_update One update at a time with Task/TaskSolver

When running an optimization on a real robot, it is convenient to seperate it in two alternating steps:

\li compute the costs from the rollouts, update the parameter distribution from the costs, and generate samples from the new distribution. The input to this step are the rollouts, which should 
contain all the variables relevant to computing the costs. The output is the new samples.

\li perform the roll-outs, given the samples in parameter space. Save all the cost-relevant variables to file.

The first part has been implemented in Python (run_one_update.py), and the second part is specific to your robot, and can be implemented however you want.

In practice, it is not convenient to run above two phases in a loop, but rather perform them step by step. That means the optimization can be stopped at any time, which is useful if you are running long optimization (and want to go to lunch or something). To enable this update-by-update approach, the loop is modified as follows:

\code

int n_dim = 2; // Optimize 2D problem

// This is the cost function to be optimized
CostFunction* cost_function = new CostFunctionQuadratic(VectorXd::Zero(n_dim));

// This is the updater which will update the distribution
double eliteness = 10.0;
Updater* updater = new UpdaterMean(eliteness);

// Some variables
MatrixXd samples;
VectorXd costs;

// DETERMINE WHICH UPDATE BY READING FROM FILE

if (i_update==0) 
{
  // This is the initial distribution, only used before the first update
  DistributionGaussian* distribution = new DistributionGaussian(VectorXd::Random(n_dim),MatrixXd::Identity(n_dim)) 
}
else 
{
    READ THE COST-RELEVANT VARIABLES FROM FILE
  
    // 2B. Evaluate the samples
    task->evaluate(cost_vars,costs);
  
    // 3. Update parameters
    updater->updateDistribution(*distribution, samples, costs, *distribution);
    
}

// 1. Sample from distribution
int n_samples_per_update = 10;
distribution->generateSamples(n_samples_per_update, samples);

// SAVE SAMPLES TO FILE. THESE SHOULD BE READ BY YOUR ROBOT

// Thus 2A. Perform the roll-outs: task_solver->performRollouts(samples,cost_vars);
// no longer occurs in the code. Your robot does this now.  

\endcode

\subsection sec_practical_howto Practical howto

\todo Update this with DMP example

The typical use case for running black-box optimization with a robot requires run_one_update.py. Here's what you need to do to make it all work.

<ul>
<li> In Python, implement a script that sets the parameters of the optimization (i.e. the initial distribution, the task, the covariance update method, etc.). 
The task should inherit from Task (see task.py), and you'll need to implement the evaluate function, which is the cost function for your task. It takes all the cost-relevant variables, and returns a cost. For an example see demo_one_update.py
</li>

<li> Implement a script "performRollouts" for your robot (in whatever language you want). The "communication protocol" between run_one_update.py and the robot is very simple (conciously made so!). Only .txt files containing matrices are exchanged. An example script (in Python) can be found in demo_fake_robot.py

<ol>

<li> The script should take a directory as an input. Suppose we are writing to the master directory "./meka_bbo/", then each update has its own directory: "./meka_bbo/update00001", "./meka_bbo/update00002", "./meka_bbo/update00003", etc.</li>

<li> Read the file "samples.txt" from the directory. The size of this file is n_samples X n_dim, where n_samples is the number of sample/rollouts, and n_dim is the dimensionality of the search space. For parallel optimization, a different file is provided for each dimension, i.e. "samples_00.txt" for the first dimension, etc.</li>

<li>Perform the n_samples rollouts on your robot, and write the resulting trajectories/forces whatever to a file "cost_vars.txt", which should be of size n_samples X n_cost_vars. Thus again, each row should represent one rollout. (For parallel optimization, there will be only one cost_vars.txt)</li>

<li>Optional: To determine the cost of the mean of the current distribution (rather than of samples from this distribution as before), read "distribution_mean", and treat it as one sample. The resulting cost-relevant variables should be written in "cost_vars_eval.txt" rather than 
"cost_vars.txt".

</ol>

</li>

<li>Run the actual optimization by alternatively calling the Python script for updating the parameters, and your script for running the rollouts.

<ul>
<li>python demo_one_update.py ./meka_bbo</li>
<li>python demo_fake_robot.py ./meka_bbo/update00001</li>
<li>python demo_one_update.py ./meka_bbo</li>
<li>python demo_fake_robot.py ./meka_bbo/update00002</li>
<li>python demo_one_update.py ./meka_bbo</li>
<li>python demo_fake_robot.py ./meka_bbo/update00003</li>
<li>python demo_one_update.py ./meka_bbo</li>
</ul>
</li>

<li>When you are done with the optimization, you can use the  function bb_plotting.plotOptimizationDir(directory) to visualize the result.
<ol>
<li>python plotOptimizationDir ./meka_bbo</li>
</ol>
</li>
</ul>


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
runOptimization() and demoOptimization.cpp):
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
  Task::evaluateRollout(cost_vars,costs)
}
\endcode

For more details, see \ref sec_dmp_bbo_task_and_task_solver

\subsection sec_bbo_task_and_task_solver CostFunction vs Task/TaskSolver

When the cost function has a simple structure, e.g. cost = \f$ x^2 \f$ it is convenient to implement the function \f$ x^2 \f$ in CostFunction::evaluate(). In robotics however, it is more suitable to make the distinction between a task (e.g. lift an object), and an entity that solves this task (e.g. your robot, my robot, a simulated robot, etc.). For these cases, the CostFunction is split into a Task and a TaskSolver, as follows:

\code
CostFunction::evaluate(samples,costs) {
  TaskSolver::performRollout(samples,cost_vars)
  Task::evaluateRollout(cost_vars,costs)
}
\endcode

For more details, see \ref sec_dmp_bbo_task_and_task_solver

 */

