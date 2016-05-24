/**
 * @file runOptimization.cpp
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

#include "dmp_bbo/runOptimizationTask.hpp"

#include <iomanip>
#include <fstream>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Core>

#include "dmp_bbo/Task.hpp"
#include "dmp_bbo/TaskSolver.hpp"
#include "dmp_bbo/Rollout.hpp"
#include "dmp_bbo/ExperimentBBO.hpp"

#include "bbo/DistributionGaussian.hpp"
#include "bbo/Updater.hpp"

#include "bbo/runOptimization.hpp" // For saving functionality

#include "dmpbbo_io/EigenFileIO.hpp"




using namespace std;
using namespace Eigen;

namespace DmpBbo {
  
bool saveToDirectory(string directory, int i_update, const DistributionGaussian& distribution, const Rollout* rollout_eval, const vector<Rollout*>& rollouts, const VectorXd& weights, const DistributionGaussian& distribution_new, bool overwrite)
{
  vector<DistributionGaussian> distribution_vec;
  distribution_vec.push_back(distribution);

  vector<DistributionGaussian> distribution_new_vec;
  distribution_new_vec.push_back(distribution_new);
  
  return saveToDirectory(directory, i_update, distribution_vec, rollout_eval, rollouts, weights, distribution_new_vec, overwrite);
}

bool saveToDirectory(string directory, int i_update, const vector<DistributionGaussian>& distribution, const Rollout* rollout_eval, const vector<Rollout*>& rollouts, const VectorXd& weights, const vector<DistributionGaussian>& distribution_new, bool overwrite)
{
  
  VectorXd cost_eval;
  if (rollout_eval!=NULL)
    rollout_eval->cost(cost_eval);
  
  MatrixXd costs(rollouts.size(),rollouts[0]->getNumberOfCostComponents());
  for (unsigned int ii=0; ii<rollouts.size(); ii++)
  {
    VectorXd cur_cost;
    rollouts[ii]->cost(cur_cost);
    costs.row(ii) = cur_cost;
  }
  
  // Save update information
  MatrixXd samples;
  saveToDirectory(directory, i_update, distribution, cost_eval, samples, costs, weights, distribution_new,overwrite);

  stringstream stream;
  stream << directory << "/update" << setw(5) << setfill('0') << i_update << "/";
  string directory_update = stream.str();
  
  // Save rollouts too
  for (unsigned int i_rollout=0; i_rollout<rollouts.size(); i_rollout++)
  {
    stringstream stream;
    stream << directory_update << "/rollout" << setw(3) << setfill('0') << i_rollout+1;
    if (!rollouts[i_rollout]->saveToDirectory(stream.str(),overwrite))
      return false;
  }
  
  if (rollout_eval!=NULL)
    if (rollout_eval->saveToDirectory(directory_update+"/rollout_eval",overwrite))
      return false;
    
  return true;    
}


void runOptimizationTask(
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
  
  int n_cost_components = task->getNumberOfCostComponents();

  // Some variables
  VectorXd sample_eval;
  MatrixXd cost_vars_eval;
  VectorXd cost_eval(1+n_cost_components);
  
  MatrixXd samples;
  MatrixXd cost_vars;
  VectorXd weights;
  MatrixXd costs(n_samples_per_update,1+n_cost_components);
  
  // tmp variables
  VectorXd total_costs(n_samples_per_update);
  VectorXd cur_cost(1+n_cost_components);
  
  // Bookkeeping
  MatrixXd learning_curve(n_updates,2+n_cost_components);
  MatrixXd exploration_curve(n_updates,2);
  
  if (save_directory.empty()) 
    cout << "init  =  " << "  distribution=" << *initial_distribution;
  
  DistributionGaussian distribution = *(initial_distribution->clone());
  DistributionGaussian distribution_new = *(initial_distribution->clone());
  
  // Optimization loop
  for (int i_update=0; i_update<n_updates; i_update++)
  {
    // 0. Get cost of current distribution mean
    sample_eval = distribution.mean().transpose();
    task_solver->performRollout(sample_eval,cost_vars_eval);
    task->evaluateRollout(cost_vars_eval,sample_eval,cost_eval);
    Rollout* rollout_eval = new Rollout(sample_eval,cost_vars_eval,cost_eval);
    
    // 1. Sample from distribution
    distribution.generateSamples(n_samples_per_update, samples);

    vector<Rollout*> rollouts(n_samples_per_update);
    for (int i_sample=0; i_sample<n_samples_per_update; i_sample++)
    {
      // 2A. Perform the rollout
      task_solver->performRollout(samples.row(i_sample),cost_vars);

      // 2B. Evaluate the rollout
      task->evaluateRollout(cost_vars,samples.row(i_sample),cur_cost);
      costs.row(i_sample) = cur_cost;

      rollouts[i_sample] = new Rollout(samples.row(i_sample),cost_vars,cur_cost);
      
    }
  
    // 3. Update parameters (first column of costs contains sum of cost components)
    total_costs = costs.col(0);
    updater->updateDistribution(distribution, samples, total_costs, weights, distribution_new);
    
    
    // Bookkeeping
    // Some output and/or saving to file (if "directory" is set)
    if (save_directory.empty()) 
    {
      cout << "\t cost_eval=" << cost_eval << endl << i_update+1 << "  " << distribution;
    }
    else
    {
      // Update learning curve
      // How many samples?
      int i_samples = i_update*n_samples_per_update;
      learning_curve(i_update,0) = i_samples;
      // Cost of evaluation
      learning_curve.block(i_update,1,1,1+n_cost_components) = cost_eval.transpose();
      
      // Exploration magnitude
      exploration_curve(i_update,0) = i_samples;
      exploration_curve(i_update,1) = sqrt(distribution.maxEigenValue()); 
      
      // Save more than just learning curve.
      if (!only_learning_curve)
      {
          saveToDirectory(save_directory,i_update,distribution,rollout_eval,rollouts,weights,distribution_new);
          if (i_update==0)
            task->savePlotRolloutScript(save_directory);
      }
    }
    
    // Distribution is new distribution
    distribution = distribution_new;
    
  }
  
  // Save learning curve to file, if necessary
  if (!save_directory.empty())
  {
    // Todo: save cost labels also
    saveMatrix(save_directory, "exploration_curve.txt",exploration_curve,overwrite);
    saveMatrix(save_directory, "learning_curve.txt",learning_curve,overwrite);
  }
}

void runOptimizationTask(ExperimentBBO* experiment, std::string save_directory, bool overwrite,   bool only_learning_curve)
{
 runOptimizationTask(
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

void runOptimizationParallelDeprecated(
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
    task->evaluateRollout(cost_vars_eval,sample_eval,cost_eval);
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
      task->evaluateRollout(cost_vars,samples.row(i_sample),cur_costs);

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
