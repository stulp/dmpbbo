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
 * \todo This function will be removed. It can be implemented much simpler. Only UpdateCovarAdapation is affected by this, and one can enforce block-diagonal-covars there (e.g. with a function setSubCovars(VectorXi). Update the docu accordingly when this is done. )
 */
void runOptimizationParallelDeprecated(Task* task, TaskSolver* task_solver, std::vector<DistributionGaussian*> distributions, Updater* updater, int n_updates, int n_samples_per_update, std::string save_directory=std::string(""),bool overwrite=false,
bool only_learning_curve=false);

bool saveToDirectory(std::string directory, int i_update, const DistributionGaussian& distribution, const Rollout* rollout_eval, const std::vector<Rollout*>& rollouts, const Eigen::VectorXd& weights, const DistributionGaussian& distribution_new, bool overwrite=false);

bool saveToDirectory(std::string directory, int i_update, const std::vector<DistributionGaussian>& distribution, const Rollout* rollout_eval, const std::vector<Rollout*>& rollouts, const Eigen::VectorXd& weights, const std::vector<DistributionGaussian>& distribution_new, bool overwrite=false);

}

#endif

/** \defgroup BBO Black Box Optimization Module
 */


/** \defgroup DMP_BBO Black Box Optimization of Dynamical Movement Primitives Module
 */

/** \page page_dmp_bbo Black Box Optimization of Dynamical Movement Primitives

This page assumes you have read the page on \ref page_bbo. When applying BBO to policy improvement (e.g. optimizing a DMP on a robot), the concept of a "rollout" becomes important. A rollout is the result of executing a policy (e.g. a DMP) with a certain set of policy parameters (e.g. the parameter of the DMP). Although the search space for optimization is in the space of the policy parameters, the costs are rather determined from the rollout.

From an implementation point of view, applying BBO to policy improvement (which may execute DMPs) requires several extensions:

\li Running multiple optimizations in parallel, one for each DOF of the DMP
\li Using rollouts with a Task/TaskSolver instead of a CostFunction
\li Allowing the optimization to be run one update at a time, instead of in a loop

\section sec_parallel_optimization Optimizing DMP DOFs in parallel

Consider a 7-DOF robotic arm. With a DMP, each DOF would be represented by a different dimension of a 7-D DMP (let as assume that each dimension of the DMP uses 10 basis functions). There are then two distinct approaches towards optimizing the parameters of the DMP with respect to a cost function:

\li Consider the search space for optimization to be 70-D, i.e. consider all open parameters of the DMP in one search space. For this you can use runOptimizationTask(), see also the demos in demoOptimizationTask.cpp and demoOptimizationDmp.cpp. 
The disadvantage is that reward-weighted averaging will require more samples to perform a robust update, especially for the covariance matrix, which is 70X70! 

\li Run a seperate optimization for each of the 7 DOFs. Thus, here we have seven 10-D search spaces (with different distributions and samples for each DOF), but using the same costs for all optimizations. This is what is proposed for instance in PI^2. This approach is not able to exploit covariance between different DOFs, but in practice it works robustly. For this you must runOptimizationParallel(), see also the demo in demoOptimizationDmpParallel.cpp. The main difference is that instead of using one distribution (mean and covariance), D distributions must be used (these are stored in a std::vector), one for each DOF.


\section sec_bbo_task_and_task_solver CostFunction vs Task/TaskSolver

When the cost function has a simple structure, e.g. cost = \f$ x^2 \f$ it is convenient to implement the function \f$ x^2 \f$ in CostFunction::evaluate(). 

In robotics however, it is suitable to make the distinction between a task (e.g. lift an object), and an entity that solves this task (e.g. your robot, my robot, a simulated robot, etc.). For these cases, the CostFunction is split into a Task and a TaskSolver, as follows:

\code
CostFunction::evaluate(samples,costs) {
  TaskSolver::performRollout(samples,cost_vars)
  Task::evaluateRollout(cost_vars,costs)
}
\endcode

The roles of the Task/TaskSolver are:

\li TaskSolver: performing the rollouts on the robot (based on the samples in policy space), and saving all the variables related to computing the costs into a file (called cost_vars.txt)

\li Task: determining the costs from the cost-relevant variables


The idea here is that the TaskSolver uses the samples to perform a rollout (e.g. the samples represent the parameters of a policy which is executed) and computes all the variables that are relevant to determining the cost (e.g. it records the forces at the robot's end-effector, if this is something that needs to be minimized)

Some further advantages of this approach:
\li Different robots can solve the exact same Task implementation of the same task.
\li Robots do not need to know about the cost function to perform rollouts (and they shouldn't)
\li The intermediate cost-relevant variables can be stored to file for visualization etc.
\li The procedures for performing the roll-outs (on-line on a robot) and doing the evaluation/updating/sampling (off-line on a computer) can be seperated, because there is a separate TaskSolver::performRollouts function.

\subsection impl Implementation

When using the Task/TaskSolver approach, the optimization process is as follows:
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
  
    for (int i_sample=0; i_sample<n_samples_per_update; i_sample++)
    {
      // 2A. Perform the rollout
      task_solver->performRollout(sample.row(i_sample),cost_vars);
  
      // 2B. Evaluate the rollout
      task->evaluateRollout(cost_vars,costs);
      costs[i_sample] = cur_costs[0];
      
    }
  
    // 3. Update parameters
    updater->updateDistribution(*distribution, samples, costs, *distribution);
    
}
\endcode

\subsubsection sec_cost_components Cost components

evaluateRollout returns a vector of costs. This is convenient for tracing different cost components. For instance, consider a task where the costs consist of going through a viapoint with minimal acceleration. In this case, the cost components are: 1) distance to the viapoint and 2) sum of acceleration. Then 
\li cost[1] = distance to the viapoint
\li cost[2] = acceleration
\li cost[0] = cost[1] + cost[2] (cost[0] must always be the sum of the individual cost components)

\subsubsection sec_cost_vars Cost-relevant variables

cost_vars should contain all variables that are relevant to computing the cost. This depends entirely on the task at hand. Whatever cost_vars contains, it should be made sure that a Task and its TaskSolver are compatible, and have the same understanding of what is contained in cost_vars. Tip: for rollouts on the robot I usually let each row in cost_vars be the relevant variables at one time step. An example is given in TaskViapoint, which implements a Task in which the first N columns in cost_vars should represent a N-D trajectory. This convention is respected by TaskSolverDmp, which is able to generate such trajectories.


\section sec_bbo_one_update One update at a time with Task/TaskSolver

\b Note: this is currently only implemented in python/dmp_bbo/

When running an optimization on a real robot, it is convenient to seperate it in two alternating steps:

\li compute the costs from the rollouts, update the parameter distribution from the costs, and generate samples from the new distribution. The input to this step are the rollouts, which should 
contain all the variables relevant to computing the costs. The output is the new samples.

\li perform the roll-outs, given the samples in parameter space. Save all the cost-relevant variables to file.

The first part has been implemented in Python (run_one_update.py), and the second part is specific to your robot, and can be implemented however you want.

In practice, it is not convenient to run above two phases in a loop, but rather perform them step by step. That means the optimization can be stopped at any time, which is useful if you are running long optimization (and want to go to lunch or something). An example of this update-by-update optimization is given in python/dmp_bbo/demos, where:

\li demo_one_update.py performs the actual optimization

\li demo_perform_rollouts.py executes the rollouts (you'll have to implement something similar on your robot)

\li demo_optimizatio_one_by_one.bash, which alternatively calls the two scripts above 

\code
#!/bin/bash

DIREC="/tmp/demo_optimization_one_by_one"

# Yes, I know there are for loops in bash ;-)
# But this makes it really explicit how to call the scripts

python demo_one_update.py $DIREC/
python demo_perform_rollouts.py $DIREC/update00001 # Replace this with your robot

python demo_one_update.py $DIREC/
python demo_perform_rollouts.py $DIREC/update00002 # Replace this with your robot

python demo_one_update.py $DIREC/
python demo_perform_rollouts.py $DIREC/update00003 # Replace this with your robot

python demo_one_update.py $DIREC/
python demo_perform_rollouts.py $DIREC/update00004 # Replace this with your robot

python demo_one_update.py $DIREC/ plotresults
\endcode

\subsection sec_practical_howto Practical howto

\todo Update this with DMP example

Here's what you need to do to make it all work.

\subsubsection sec_specify_task Specify the task

Copy demo_one_update.py to, for instance, one_update_my_task.py. In it, replace the Task with whatever your task is. The most important function is evaluateRollout(self, cost_vars), which takes the cost-relevant variables, and returns its cost. See the sections on \ref sec_cost_vars and \ref sec_cost_components for what these  should contain.

If you want functionality for plotting a rollout, you can also implement the function plotRollout(self,cost_vars,ax), which takes costs_vars and visualizes the rollout they represent on the axis 'ax'.

\subsubsection sec_specify_settings Specify the settings of the optimization

This includes the initial distribution, the method for updating the covariance matrix and the number of samples per update. This is all set in one_update_my_task.py (your copy of demo_one_update.py).

\subsubsection sec_specify_robot Enable your robot to perform rollouts

Implement a script "performRollouts" for your robot (in whatever language you want). It should take a directory as an input. From this trajectory, it should read the file "policy_parameters.txt" and execute whatever movement it makes with these policy parameters (e.g. the parameters of the DMP, which can be set with the function setParameterVectorSelected, which it inheritst from Parameterizable)

It should write the resulting rollout to a file called cost_vars.txt (in the same directory from which policy_parameters.txt was read). This should be a vector or a matrix. What it contains depends entirely on your task. So what your robot writes to cost_vars.txt should be compatible with Task.evaluateRollout(self,cost_vars).

The "communication protocol" between run_one_update.py and the robot is very simple (conciously made so!). Only .txt files containing matrices are exchanged. An example script (in Python) can be found in demo_perform_rollouts.py. Again, this script can be anything you want (Python, bash script, Matlab, real robot), as long as it respects the format of cost_vars.txt that Task.evaluateRollout() expects.





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
 

