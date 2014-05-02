/**
 * @file runEvolutionaryOptimizationParallel.cpp
 * @brief  Source file for function to run multiple evolutionary optimization processes in parallel.
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
#include "dmp_bbo/runEvolutionaryOptimizationParallel.hpp"

#include "bbo/Task.hpp"
#include "bbo/DistributionGaussian.hpp"
#include "bbo/Updater.hpp"
#include "dmp_bbo/TaskSolverParallel.hpp"
#include "dmp_bbo/UpdateSummaryParallel.hpp"
#include "dmpbbo_io/EigenFileIO.hpp"

#include <iomanip>
#include <fstream>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

namespace DmpBbo {

bool saveTask(string save_directory, Task* task, TaskSolver* task_solver, bool overwrite=false);

bool saveUpdate(string save_directory, int i_update, const vector<UpdateSummary>& update_summaries, const MatrixXd& cost_vars, const MatrixXd& cost_vars_eval, bool overwrite=false);

void runEvolutionaryOptimizationParallel(Task* task, TaskSolverParallel* task_solver, vector<DistributionGaussian*> distributions, Updater* updater, int n_updates, int n_samples_per_update, string save_directory, bool overwrite, bool only_learning_curve)
{  
  // Some variables
  int n_parallel = distributions.size();
  vector<MatrixXd> sample_eval(n_parallel);
  vector<MatrixXd> samples(n_parallel);
  MatrixXd cost_vars, cost_vars_eval;
  VectorXd costs, cost_eval;
  // Bookkeeping
  // This is where the UpdateSummary objects are stored, one for each parallel optimization
  vector<UpdateSummary> cur_summaries(n_parallel);
  // This is the UpdateSummaryParallel, that merges the above
  UpdateSummaryParallel update_summary;
  update_summary.distributions.resize(n_parallel); // Pre-allocate enough space here
  update_summary.distributions_new.resize(n_parallel); // Pre-allocate enough space here
  update_summary.samples.resize(n_parallel); // Pre-allocate enough space here
  // This is where the UpdateSummaryParallel objects are accumulated, one for each update
  vector<UpdateSummaryParallel> update_summaries;
  
  // Optimization loop
  for (int i_update=0; i_update<n_updates; i_update++)
  {
    // 0. Get cost of current distribution mean
    for (int pp=0; pp<n_parallel; pp++)
      sample_eval[pp] = distributions[pp]->mean().transpose();
    task_solver->performRollouts(sample_eval,cost_vars_eval);
    task->evaluate(cost_vars_eval,cost_eval);
    
    // 1. Sample from distribution
    for (int pp=0; pp<n_parallel; pp++)
      distributions[pp]->generateSamples(n_samples_per_update, samples[pp]);

    // 2. Perform rollouts for the samples
    task_solver->performRollouts(samples, cost_vars);
      
    // 3. Evaluate the last batch of rollouts
    task->evaluate(cost_vars,costs);
  
    // 4. Update parameters
    for (int pp=0; pp<n_parallel; pp++)
      updater->updateDistribution(*(distributions[pp]), samples[pp], costs, *(distributions[pp]),
                                                                      (cur_summaries[pp]));
      
    // Some output and/or saving to file (if "directory" is set)
    if (save_directory.empty()) 
    {
      cout << i_update+1 << "  cost_eval=" << cost_eval << "  ";
      if (distributions.size()==1)
        cout << *(distributions[0]);
      else
        for (unsigned int ii=0; ii<distributions.size(); ii++)
          cout  << endl << "       distributions["<<ii<<"]=" << *(distributions[ii]);
      cout << endl;
    }
    else
    {
      update_summary.cost_eval = cost_eval[0];
      update_summary.cost_vars_eval = cost_vars_eval;
      update_summary.cost_vars = cost_vars;
      update_summary.costs =  costs;
      for (int pp=0; pp<n_parallel; pp++)
      {
        update_summary.distributions[pp] =  cur_summaries[pp].distribution;
        update_summary.distributions_new[pp] =  cur_summaries[pp].distribution_new;
        update_summary.samples[pp] =  cur_summaries[pp].samples;
      }
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

}
