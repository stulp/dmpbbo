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

#include "bbo/runEvolutionaryOptimization.hpp"

#include "bbo/Task.hpp"
#include "dmp_bbo/TaskSolverParallel.hpp"
#include "bbo/DistributionGaussian.hpp"
#include "bbo/Updater.hpp"
#include "bbo/UpdateSummary.hpp"
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

void runEvolutionaryOptimizationParallel(Task* task, TaskSolverParallel* task_solver, vector<DistributionGaussian*> distributions, Updater* updater, int n_updates, int n_samples_per_update, string save_directory, bool overwrite)
{
  if (!save_directory.empty()) 
    task->savePerformRolloutsPlotScript(save_directory);
  
  if (!saveTask(save_directory, task, task_solver, overwrite))
    return;

  // Some definitions
  int n_parallel = distributions.size();
  vector<MatrixXd> sample_eval(n_parallel);
  vector<MatrixXd> samples(n_parallel);
  MatrixXd cost_vars, cost_vars_eval;
  VectorXd costs, cost_eval;
  vector<UpdateSummary> update_summaries(n_parallel);
  
  // Optimization loop
  //cout << "init  =  " << "  distribution[0]=" << *(distributions[0] << endl;
  for (int i_update=1; i_update<=n_updates; i_update++)
  {
    // 0. Get cost of current distribution mean
    for (int pp=0; pp<n_parallel; pp++)
      sample_eval[pp] = distributions[pp]->mean().transpose();
    task_solver->performRollouts(sample_eval,cost_vars_eval);
    task->evaluate(cost_vars_eval,cost_eval);
    for (int pp=0; pp<n_parallel; pp++)
      update_summaries[pp].cost_eval = cost_eval[0];
    
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
                                                                      (update_summaries[pp]));
      
    // Some output and saving to file (if "directory" is set)
    cout << "update=" << i_update << "  mean(costs)=" << costs.mean() << "    ";
    if (distributions.size()==1)
      cout << *(distributions[0]);
    else
      for (unsigned int ii=0; ii<distributions.size(); ii++)
        cout  << endl << "    distributions["<<ii<<"]=" << *(distributions[ii]);
    cout << endl;
    
    if (!save_directory.empty())
    {
      saveUpdate(save_directory,i_update,update_summaries,cost_vars,cost_vars_eval,overwrite);
      if (i_update==1) 
        task->savePerformRolloutsPlotScript(save_directory);
    }
    
  }
}

bool saveUpdate(string save_directory, int i_update, const vector<UpdateSummary>& update_summaries, const MatrixXd& cost_vars, const MatrixXd& cost_vars_eval, bool overwrite)
{
  if (save_directory.empty())
    return true;

  stringstream stream;
  stream << save_directory << "/update" << setw(5) << setfill('0') << i_update << "/";
  string directory_update = stream.str();
  
  int n_parallel = update_summaries.size();
  if (n_parallel==1)
  {
    saveToDirectory(update_summaries[0], directory_update);
  }
  else
  {
    for (int i_parallel=0; i_parallel<n_parallel; i_parallel++)
    {
      stringstream stream_dim;
      stream_dim << directory_update << "/dim" << setw(2) << setfill('0') << i_parallel << "/";
      saveToDirectory(update_summaries[i_parallel], stream_dim.str());
    }
    VectorXi n_parallel_vec = VectorXi::Constant(1,n_parallel);
    saveMatrix(save_directory,"n_parallel.txt",n_parallel_vec,overwrite);
  }

  saveMatrix(directory_update,"cost_vars_eval.txt",cost_vars_eval,overwrite);
  VectorXd cost_eval = VectorXd::Constant(1,update_summaries[0].cost_eval);
  saveMatrix(directory_update,"cost_eval.txt",cost_eval,overwrite);

  saveMatrix(directory_update,"cost_vars.txt",cost_vars,overwrite);
  saveMatrix(directory_update,"costs.txt",update_summaries[0].costs,overwrite);

  saveMatrix(directory_update,"weights.txt",update_summaries[0].weights,overwrite);
  
  return true;
 
}



bool saveTask(string save_directory, Task* task, TaskSolver* task_solver, bool overwrite)
{
  // todo This needs to be replaced with boost serialization
  if (save_directory.empty())
    return true;

  if (boost::filesystem::exists(save_directory))
  {
    if (overwrite==false)
    {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "ERROR: Trying to save data to file, but directory '" << save_directory << "' already exists. ABORT." << endl;
      return false;
    }
  }
  else
  {
    // Make directory if it doesn't already exist
    if (!boost::filesystem::create_directories(save_directory))
    {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "Couldn't make directory file '" << save_directory << "'." << endl;
      return false;
    }      
  }
  

  string filename = save_directory+"/task.txt";
  ofstream file;
  file.open(filename.c_str());
  if (!file.is_open())
  {
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "Couldn't open file '" << filename << "' for writing." << endl;
    return false;
  }
  file << *task;
  file.close();

  filename = save_directory+"/task_solver.txt";
  file.open(filename.c_str());
  if (!file.is_open())
  {
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "Couldn't open file '" << filename << "' for writing." << endl;
    return false;
  }
  file << *task_solver;
  file.close();
     
  return true;
}

}
