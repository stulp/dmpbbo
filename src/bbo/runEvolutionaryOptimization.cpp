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
#include "bbo/CostFunction.hpp"
#include "bbo/Task.hpp"
#include "bbo/TaskSolver.hpp"
#include "bbo/Rollout.hpp"
#include "bbo/ExperimentBBO.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"




using namespace std;
using namespace Eigen;

namespace DmpBbo {
  
bool saveToDirectory(string directory, int i_update, const vector<DistributionGaussian>& distributions, double* cost_eval, const MatrixXd& samples, const VectorXd& costs, const VectorXd& weights, const vector<DistributionGaussian>& distributions_new, bool overwrite)
{
  // Make directory if it doesn't already exist
  if (!boost::filesystem::exists(directory))
  {
    if (!boost::filesystem::create_directories(directory))
    {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "Couldn't make directory file '" << directory << "'." << endl;
      return false;
    }
  }
  
  stringstream stream;
  stream << directory << "/update" << setw(5) << setfill('0') << i_update;
  string dir_update = stream.str();
  
  // Make directory if it doesn't already exist
  if (!boost::filesystem::exists(dir_update))
  {
    if (!boost::filesystem::create_directories(dir_update))
    {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "Couldn't make directory file '" << dir_update << "'." << endl;
      return false;
    }
  }

  // Abbreviations to make it fit on one line
  bool ow = overwrite;
  string dir = dir_update;
  
  assert(distributions.size()==distributions_new.size());
  
  if (distributions.size()==1)
  {
    if (!saveMatrix(dir, "distribution_mean.txt",      distributions[0].mean(),  ow)) return false;
    if (!saveMatrix(dir, "distribution_covar.txt",     distributions[0].covar(), ow)) return false;
  }
  else
  {
    for (unsigned int dd=0; dd<distributions.size(); dd++)
    { 
      stringstream stream;
      stream  << "/distribution_new_" << setw(3) << setfill('0') << dd;
      if (!saveMatrix(dir, stream.str()+"_mean.txt", distributions_new[dd].mean(), ow)) 
        return false;
      if (!saveMatrix(dir, stream.str()+"_covar.txt", distributions_new[dd].covar(), ow)) 
        return false;
      
    }
  }
  
  if (cost_eval!=NULL)
  {
    VectorXd cost_eval_vec = VectorXd::Constant(1,*cost_eval);
    if (!saveMatrix(dir, "cost_eval.txt",            cost_eval_vec,          ow)) return false;
  }

  if (samples.size()>0)
    if (!saveMatrix(dir, "samples.txt",              samples,                ow)) return false;
  if (costs.size()>0)
    if (!saveMatrix(dir, "costs.txt",                costs,                  ow)) return false;
  if (weights.size()>0)
    if (!saveMatrix(dir, "weights.txt",              weights,                ow)) return false;
  if (distributions.size()==1)
  {
    if (!saveMatrix(dir, "distribution_new_mean.txt",  distributions_new[0].mean(),  ow)) return false;
    if (!saveMatrix(dir, "distribution_new_covar.txt", distributions_new[0].covar(), ow)) return false;
  }
  else
  {
    for (unsigned int dd=0; dd<distributions.size(); dd++)
    { 
      stringstream stream;
      stream  << "/distribution_new_" << setw(3) << setfill('0') << dd;
      if (!saveMatrix(dir, stream.str()+"_mean.txt", distributions_new[dd].mean(), ow)) 
        return false;
      if (!saveMatrix(dir, stream.str()+"_covar.txt", distributions_new[dd].covar(), ow)) 
        return false;
      
    }
  }
  return true;    
}

bool saveToDirectory(string directory, int i_update, const DistributionGaussian& distribution, double* cost_eval, const MatrixXd& samples, const VectorXd& costs, const VectorXd& weights, const DistributionGaussian& distribution_new, bool overwrite)
{
  vector<DistributionGaussian> distribution_vec;
  distribution_vec.push_back(distribution);

  vector<DistributionGaussian> distribution_new_vec;
  distribution_new_vec.push_back(distribution_new);
  
  return saveToDirectory(directory, i_update, distribution_vec, cost_eval, samples, costs, weights, distribution_new_vec, overwrite);
}

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
  double cost_eval;
  MatrixXd samples;
  VectorXd sample;
  VectorXd weights;
  VectorXd costs(n_samples_per_update);
  
  // Bookkeeping
  MatrixXd learning_curve(n_updates,3);
  
  if (save_directory.empty()) 
    cout << "init  =  " << "  distribution=" << *initial_distribution;
  
  DistributionGaussian distribution = *(initial_distribution->clone());
  DistributionGaussian distribution_new = *(initial_distribution->clone());
  
  // Optimization loop
  for (int i_update=0; i_update<n_updates; i_update++)
  {
    // 0. Get cost of current distribution mean
    cost_eval = cost_function->evaluate(distribution.mean().transpose());
    
    // 1. Sample from distribution
    distribution.generateSamples(n_samples_per_update, samples);
      
    // 2. Evaluate the samples
    for (int i_sample=0; i_sample<n_samples_per_update; i_sample++)
      costs[i_sample] = cost_function->evaluate(samples.row(i_sample));
  
    // 3. Update parameters
    updater->updateDistribution(distribution, samples, costs, weights, distribution_new);
    
    
    // Bookkeeping
    // Some output and/or saving to file (if "directory" is set)
    if (save_directory.empty()) 
    {
      cout << "\t cost_eval=" << cost_eval << endl << i_update+1 << "  " << distribution;
    }
    else
    {
      // Update learning curve
      learning_curve(i_update,0) = i_update*n_samples_per_update; // How many samples so far?
      learning_curve(i_update,1) = cost_eval;                     // Cost of evaluation
      learning_curve(i_update,2) = sqrt(distribution.maxEigenValue()); // Exploration magnitude
      // Save more than just learning curve.
      if (!only_learning_curve)
      {
        saveToDirectory(save_directory, i_update, distribution, &cost_eval, samples, costs, weights, distribution_new);
      }
    }
    
    // Distribution is new distribution
    distribution = distribution_new;
    
  }
  
  // Save learning curve to file, if necessary
  if (!save_directory.empty())
    saveMatrix(save_directory, "learning_curve.txt",learning_curve,overwrite);

}

bool saveToDirectory(string directory, int i_update, const vector<DistributionGaussian>& distribution, const Rollout* rollout_eval, const vector<Rollout*>& rollouts, const VectorXd& weights, const vector<DistributionGaussian>& distribution_new, bool overwrite)
{
  
  double* cost_eval = NULL;
  if (rollout_eval!=NULL)
  {
    double c = rollout_eval->total_cost();
    cost_eval = &c;
  }
  
  VectorXd costs(rollouts.size());
  for (unsigned int ii=0; ii<rollouts.size(); ii++)
    costs[ii] = rollouts[ii]->total_cost();

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

bool saveToDirectory(string directory, int i_update, const DistributionGaussian& distribution, const Rollout* rollout_eval, const vector<Rollout*>& rollouts, const VectorXd& weights, const DistributionGaussian& distribution_new, bool overwrite)
{
  vector<DistributionGaussian> distribution_vec;
  distribution_vec.push_back(distribution);

  vector<DistributionGaussian> distribution_new_vec;
  distribution_new_vec.push_back(distribution_new);
  
  return saveToDirectory(directory, i_update, distribution_vec, rollout_eval, rollouts, weights, distribution_new_vec, overwrite);
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
  VectorXd sample_eval;
  VectorXd cost_eval;
  MatrixXd cost_vars_eval;

  MatrixXd samples;
  MatrixXd cost_vars;
  VectorXd cur_costs;
  VectorXd total_costs(n_samples_per_update);
  
  VectorXd weights;

  // Bookkeeping
  MatrixXd learning_curve(n_updates,3);
  
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
    task->evaluate(cost_vars_eval,cost_eval);
    Rollout* rollout_eval = new Rollout(sample_eval,cost_vars_eval,cost_eval);
    
    // 1. Sample from distribution
    distribution.generateSamples(n_samples_per_update, samples);

    vector<Rollout*> rollouts(n_samples_per_update);
    for (int i_sample=0; i_sample<n_samples_per_update; i_sample++)
    {
      // 2A. Perform the roll-outs
      task_solver->performRollout(samples.row(i_sample),cost_vars);

      // 2B. Evaluate the samples
      task->evaluate(cost_vars,cur_costs);
      total_costs[i_sample] = cur_costs[0];

      rollouts[i_sample] = new Rollout(samples.row(i_sample),cost_vars,cur_costs);
      
    }
    
  
    // 3. Update parameters (first column of costs contains sum of cost components)
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
      learning_curve(i_update,0) = i_update*n_samples_per_update;      // How many samples so far?
      learning_curve(i_update,1) = cost_eval[0];                       // Cost of evaluation
      learning_curve(i_update,2) = sqrt(distribution.maxEigenValue()); // Exploration magnitude
      // Save more than just learning curve. 
      if (!only_learning_curve)
      {
          saveToDirectory(save_directory,i_update,distribution,rollout_eval,rollouts,weights,distribution_new);
          if (i_update==0)
            task->savePerformRolloutsPlotScript(save_directory);
      }
    }
    
    // Distribution is new distribution
    distribution = distribution_new;
    
  }
  
  // Save learning curve to file, if necessary
  if (!save_directory.empty())
    saveMatrix(save_directory, "learning_curve.txt",learning_curve,overwrite);
  
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
     overwrite);
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
