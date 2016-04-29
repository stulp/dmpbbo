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

#include "bbo/runOptimization.hpp"

#include <iomanip>
#include <fstream>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Core>

#include "bbo/DistributionGaussian.hpp"
#include "bbo/Updater.hpp"
#include "bbo/CostFunction.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"




using namespace std;
using namespace Eigen;

namespace DmpBbo {

bool saveToDirectory(
  string directory, 
  int i_update, 
  const DistributionGaussian& distribution, 
  double* cost_eval, 
  const Eigen::MatrixXd& samples, 
  const Eigen::VectorXd& costs, 
  const Eigen::VectorXd& weights, 
  const DistributionGaussian& distribution_new, 
  bool overwrite)
{
  vector<DistributionGaussian> distribution_vec;
  distribution_vec.push_back(distribution);

  vector<DistributionGaussian> distribution_new_vec;
  distribution_new_vec.push_back(distribution_new);
  
  return saveToDirectory(directory, i_update, distribution_vec, cost_eval, samples, costs, weights, distribution_new_vec, overwrite);
}

bool saveToDirectory(
  std::string directory, 
  int i_update, 
  const std::vector<DistributionGaussian>& distributions, 
  double* cost_eval, 
  const Eigen::MatrixXd& samples, 
  const Eigen::VectorXd& costs, 
  const Eigen::VectorXd& weights, 
  const std::vector<DistributionGaussian>& distributions_new, bool overwrite)
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
  int n_parallel = distributions.size();
  
  if (n_parallel>1)
  { 
    VectorXi covar_block_sizes(n_parallel);
    for (int pp=0; pp<n_parallel; pp++)
      covar_block_sizes[pp] = distributions[pp].mean().size();
    if (!saveMatrix(dir, "covar_block_sizes.txt",covar_block_sizes,ow)) return false;
  }
  
  VectorXi offsets(n_parallel+1);
  offsets[0] = 0;
  for (int ii=0; ii<n_parallel; ii++)
    offsets[ii+1] = offsets[ii] + distributions[ii].mean().size();
  int sum_n_dims = offsets[n_parallel];
  
  VectorXd mean_merged = VectorXd::Zero(sum_n_dims);
  MatrixXd covar_merged = MatrixXd::Zero(sum_n_dims,sum_n_dims);
  for (int pp=0; pp<n_parallel; pp++)
  {
    int offset = offsets[pp];
    int width = offsets[pp+1]-offsets[pp];
    mean_merged.segment(offset,width) = distributions[pp].mean().transpose();
    covar_merged.block(offset,offset,width,width) = distributions[pp].covar();
  }
  
  
  if (!saveMatrix(dir, "distribution_mean.txt", mean_merged,  ow)) return false;
  if (!saveMatrix(dir, "distribution_covar.txt",covar_merged, ow)) return false;

  for (int pp=0; pp<n_parallel; pp++)
  {
    int offset = offsets[pp];
    int width = offsets[pp+1]-offsets[pp];
    mean_merged.segment(offset,width) = distributions_new[pp].mean().transpose();
    covar_merged.block(offset,offset,width,width) = distributions_new[pp].covar();
  }
  
  
  if (!saveMatrix(dir, "distribution_new_mean.txt", mean_merged,  ow)) return false;
  if (!saveMatrix(dir, "distribution_new_covar.txt",covar_merged, ow)) return false;
  
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
  return true;    
}


void runOptimization(
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



}
