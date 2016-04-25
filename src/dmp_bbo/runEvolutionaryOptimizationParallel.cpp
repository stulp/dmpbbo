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

#include "dmp_bbo/runEvolutionaryOptimizationParallel.hpp"

#include "bbo/runEvolutionaryOptimization.hpp"

#include "bbo/Task.hpp"
#include "bbo/TaskSolver.hpp"
#include "bbo/DistributionGaussian.hpp"
#include "bbo/Updater.hpp"
#include "bbo/Rollout.hpp"
#include "dmpbbo_io/EigenFileIO.hpp"

#include <iomanip>
#include <fstream>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

namespace DmpBbo {

//bool saveTask(string save_directory, Task* task, TaskSolver* task_solver, bool overwrite=false);

//bool saveUpdate(string save_directory, int i_update, const vector<UpdateSummary>& update_summaries, const MatrixXd& cost_vars, const MatrixXd& cost_vars_eval, bool overwrite=false);

/*
MatrixXd convertSamples(vector<MatrixXd> samples)
{
  // Input: n_parallel X n_samples X n_dims
  // Output: n_samples X (n_parallel * n_dims)
  int n_parallel = samples.size();
  int n_samples = samples[0].size();
  int n_cols = 0;
  for (int ii=0; ii<n_parallel; ii++)
    n_cols += samples[n_parallel].cols();
  
  MatrixXd converted_samples(n_samples,n_cols);
  return 0;
}
*/

void runEvolutionaryOptimizationParallel(
  Task* task, 
  TaskSolver* task_solver, 
  vector<DistributionGaussian*> initial_distributions, 
  Updater* updater, 
  int n_updates, 
  int n_samples_per_update, 
  string save_directory, 
  bool overwrite, 
  bool only_learning_curve)
{  
  // Some variables
  int n_parallel = initial_distributions.size();
  assert(n_parallel>=2);
  
  int n_samples = n_samples_per_update; // Shorthand
  
  VectorXi offsets(n_parallel+1);
  offsets[0] = 0;
  for (int ii=0; ii<n_parallel; ii++)
    offsets[ii+1] = offsets[ii] + initial_distributions[ii]->mean().size();
  int sum_n_dims = offsets[n_parallel];
  
  // n_parallel X n_samples X n_dims
  // Note: n_samples must be the same for all, n_dims varies
  //vector<MatrixXd> sample(n_parallel);
  //for (int ii=0; ii<n_parallel; ii++)
  //  // Pre-allocate memory just to be clear.
  //  sample[ii] = MatrixXd(n_samples_per_update,initial_distributions[ii]->mean().size());

  MatrixXd samples(n_samples,sum_n_dims);
  
  
  // Some variables
  VectorXd sample_eval(sum_n_dims);
  VectorXd cost_eval;
  MatrixXd cost_vars_eval;

  MatrixXd samples_per_parallel;
  MatrixXd cost_vars;
  VectorXd cur_costs;
  VectorXd costs(n_samples);
  VectorXd total_costs(n_samples);
  
  VectorXd weights;
  
  // Bookkeeping
  MatrixXd learning_curve(n_updates,3);
  
  vector<DistributionGaussian> distributions;
  vector<DistributionGaussian> distributions_new;
  for (int ii=0; ii<n_parallel; ii++)
  {
    distributions.push_back(*(initial_distributions[ii]->clone()));
    distributions_new.push_back(*(initial_distributions[ii]->clone()));
  }
  
  // Optimization loop
  for (int i_update=0; i_update<n_updates; i_update++)
  {
    
    // 0. Get cost of current distribution mean
    for (int pp=0; pp<n_parallel; pp++)
      sample_eval.segment(offsets[pp],offsets[pp+1]-offsets[pp]) = distributions[pp].mean().transpose();
    task_solver->performRollout(sample_eval,cost_vars_eval);
    task->evaluateRollout(cost_vars_eval,cost_eval);
    Rollout* rollout_eval = new Rollout(sample_eval,cost_vars_eval,cost_eval);
    
    // 1. Sample from distribution
    for (int pp=0; pp<n_parallel; pp++)
    {
      distributions[pp].generateSamples(n_samples, samples_per_parallel);
      int width = offsets[pp+1]-offsets[pp];
      samples.block(0,offsets[pp],n_samples,width) = samples_per_parallel;
    }
      
    vector<Rollout*> rollouts(n_samples_per_update);
    for (int i_sample=0; i_sample<n_samples_per_update; i_sample++)
    {
        
      // 2. Perform rollouts for the samples
      task_solver->performRollout(samples.row(i_sample), cost_vars);
      
      // 3. Evaluate the last batch of rollouts
      task->evaluateRollout(cost_vars,cur_costs);

      // Bookkeeping
      costs[i_sample] = cur_costs[0];
      rollouts[i_sample] = new Rollout(samples.row(i_sample),cost_vars,cur_costs);
      
    }
  
    // 4. Update parameters
    for (int pp=0; pp<n_parallel; pp++)
    {
      int width = offsets[pp+1]-offsets[pp];
      samples_per_parallel = samples.block(0,offsets[pp],n_samples,width);
      updater->updateDistribution(distributions[pp], samples_per_parallel, costs, weights, distributions_new[pp]);
    }
      
    // Some output and/or saving to file (if "directory" is set)
    if (save_directory.empty()) 
    {
      cout << i_update+1 << "  cost_eval=" << cost_eval << endl;
    }
    else
    {
      // Update learning curve
      // How many samples so far?
      learning_curve(i_update,0) = i_update*n_samples_per_update;      
      // Cost of evaluation
      learning_curve(i_update,1) = cost_eval[0];
      // Exploration magnitude
      learning_curve(i_update,2) = 0.0;
      for (int pp=0; pp<n_parallel; pp++)
        learning_curve(i_update,2) += sqrt(distributions[pp].maxEigenValue()); 
      // Save more than just learning curve. 
      if (!only_learning_curve)
      {
          saveToDirectory(save_directory,i_update,distributions,rollout_eval,rollouts,weights,distributions_new);
          if (i_update==0)
            task->savePlotRolloutScript(save_directory);
      }
    }
    
    // Distribution is new distribution
    for (int ii=0; ii<n_parallel; ii++)
      distributions[ii] = distributions_new[ii];
  }
  
}

}
