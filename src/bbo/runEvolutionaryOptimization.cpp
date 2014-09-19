/**
 * @file runEvolutionaryOptimization.cpp
 * @brief  Source file for function to run an evolutionary optimization process.
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

#include "bbo/runEvolutionaryOptimization.hpp"

#include <iomanip>
#include <fstream>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Core>

#include "bbo/DistributionGaussian.hpp"
#include "bbo/Updater.hpp"
#include "bbo/UpdateSummary.hpp"
#include "bbo/CostFunction.hpp"
#include "bbo/Task.hpp"
#include "bbo/TaskSolver.hpp"
#include "bbo/ExperimentBBO.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"




using namespace std;
using namespace Eigen;

namespace DmpBbo {

void runEvolutionaryOptimization(
  const CostFunction* const cost_function, 
  const DistributionGaussian* const initial_distribution, 
  const Updater* const updater, 
  int n_updates, 
  int n_samples_per_update, 
  std::string save_directory,
  bool overwrite,
  bool only_learning_curve)
{

  // Some variables
  MatrixXd samples;
  VectorXd costs, cost_eval;
  // Bookkeeping
  vector<UpdateSummary> update_summaries;
  UpdateSummary update_summary;
  
  if (save_directory.empty()) 
    cout << "init  =  " << "  distribution=" << *initial_distribution;
  
  // Optimization loop
  DistributionGaussian* distribution = initial_distribution->clone();
  for (int i_update=0; i_update<n_updates; i_update++)
  {
    // 0. Get cost of current distribution mean
    cost_function->evaluate(distribution->mean().transpose(),cost_eval);
    
    // 1. Sample from distribution
    distribution->generateSamples(n_samples_per_update, samples);

    // 2. Evaluate the samples
    cost_function->evaluate(samples,costs);
  
    // 3. Update parameters
    updater->updateDistribution(*distribution, samples, costs, *distribution, update_summary);
      
    
    // Some output and/or saving to file (if "directory" is set)
    if (save_directory.empty()) 
    {
      cout << "\t cost_eval=" << cost_eval << endl << i_update+1 << "  " << *distribution;
    }
    else
    {
      update_summary.cost_eval = cost_eval[0];
      update_summaries.push_back(update_summary);
    }
  }
  
  // Save update summaries to file, if necessary
  if (!save_directory.empty())
    saveToDirectory(update_summaries,save_directory,overwrite,only_learning_curve);

}

// This function could have been integrated with the above. But I preferred to duplicate a bit of
// code so that the difference between running an optimziation with a CostFunction or
// Task/TaskSolver is more apparent.
void runEvolutionaryOptimization(
  const Task* const task, 
  const TaskSolver* const task_solver, 
  const DistributionGaussian* const initial_distribution, 
  const Updater* const updater, 
  int n_updates, 
  int n_samples_per_update, 
  std::string save_directory, 
  bool overwrite,
  bool only_learning_curve)
{
  // Some variables
  MatrixXd samples;
  MatrixXd cost_vars, cost_vars_eval;
  VectorXd costs, cost_eval;
  // Bookkeeping
  vector<UpdateSummary> update_summaries;
  UpdateSummary update_summary;
  
  if (save_directory.empty()) 
    cout << "init  =  " << "  distribution=" << *initial_distribution;
  
  // Optimization loop
  DistributionGaussian* distribution = initial_distribution->clone();
  for (int i_update=0; i_update<n_updates; i_update++)
  {
    // 0. Get cost of current distribution mean
    task_solver->performRollouts(distribution->mean().transpose(),cost_vars_eval);
    task->evaluate(cost_vars_eval,cost_eval);
    
    // 1. Sample from distribution
    distribution->generateSamples(n_samples_per_update, samples);

    // 2A. Perform the roll-outs
    task_solver->performRollouts(samples,cost_vars);
  
    // 2B. Evaluate the samples
    task->evaluate(cost_vars,costs);
  
    // 3. Update parameters
    updater->updateDistribution(*distribution, samples, costs, *distribution, update_summary);
      
    
    // Some output and/or saving to file (if "directory" is set)
    if (save_directory.empty()) 
    {
      cout << "\t cost_eval=" << cost_eval << endl << i_update+1 << "  " << *distribution;
    }
    else
    {
      update_summary.cost_eval = cost_eval[0];
      update_summary.cost_vars_eval = cost_vars_eval;
      update_summary.cost_vars = cost_vars;
      update_summaries.push_back(update_summary);
    }
  }
  
  // Save update summaries to file, if necessary
  if (!save_directory.empty())
  {
    saveToDirectory(update_summaries,save_directory,overwrite,only_learning_curve);
    // If you store only the learning curve, no need to save the script to visualize rollouts    
    if (!only_learning_curve)
      task->savePerformRolloutsPlotScript(save_directory);
  }

}

void runEvolutionaryOptimization(ExperimentBBO* experiment, std::string save_directory, bool overwrite,   bool only_learning_curve)
{
 if (experiment->cost_function!=NULL)
 {
   runEvolutionaryOptimization(
     experiment->cost_function,
     experiment->initial_distribution,
     experiment->updater, 
     experiment->n_updates, 
     experiment->n_samples_per_update, 
     save_directory,
     overwrite,
     only_learning_curve);
 }
 else
 {
   runEvolutionaryOptimization(
     experiment->task,
     experiment->task_solver,
     experiment->initial_distribution,
     experiment->updater, 
     experiment->n_updates, 
     experiment->n_samples_per_update, 
     save_directory,
     overwrite,
     only_learning_curve);
 }
}


}
