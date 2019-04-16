/**
 * @file   UpdaterMean.cpp
 * @brief  UpdaterMean class source file.
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

#include "bbo/updaters/UpdaterMean.hpp"

#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

namespace DmpBbo {

UpdaterMean::UpdaterMean(double eliteness, std::string weighting_method)
: eliteness_(eliteness), weighting_method_(weighting_method)
{
}

void UpdaterMean::updateDistribution(const DistributionGaussian& distribution, const MatrixXd& samples, const VectorXd& costs, VectorXd& weights, DistributionGaussian& distribution_new) const
{
  VectorXd mean_new;
  updateDistributionMean(distribution.mean(), samples, costs, weights, mean_new);

  distribution_new.set_mean(mean_new);
  distribution_new.set_covar(distribution.covar());
}

void UpdaterMean::updateDistributionMean(const VectorXd& mean, const MatrixXd& samples, const VectorXd& costs, VectorXd& weights, VectorXd& mean_new) const
{
  costsToWeights(costs,weighting_method_,eliteness_,weights);

  // Compute new mean with reward-weighed averaging
  // mean    = 1 x n_dims
  // weights = 1 x n_samples
  // samples = n_samples x n_dims
  mean_new = weights.transpose()*samples;
  /*
  cout << "  mean=" << mean << endl;
  cout << "  weights=" << weights << endl;
  cout << "  samples=" << samples << endl;
  cout << "  mean_new=" << mean_new << endl;
  */
}


void  UpdaterMean::costsToWeights(const VectorXd& costs, string weighting_method, double eliteness, VectorXd& weights) const
{
  weights.resize(costs.size());
  if (weighting_method.compare("PI-BB")==0)
  {
    // PI^2 style weighting: continuous, cost exponention
    double h = eliteness; // In PI^2, eliteness parameter is known as "h"
    double range = costs.maxCoeff()-costs.minCoeff();
    if (range==0)
      weights.fill(1);
    else
      weights = (-h*(costs.array()-costs.minCoeff())/range).exp();
  } 
  else if (weighting_method.compare("CMA-ES")==0 || weighting_method.compare("CEM")==0 )
  {
    // CMA-ES and CEM are rank-based, so we must first sort the costs, and the assign a weight to 
    // each rank.
    VectorXd costs_sorted = costs; 
    std::sort(costs_sorted.data(), costs_sorted.data()+costs_sorted.size());
    // In Python this is more elegant because we have argsort.
    // indices = np.argsort(costs)
    // It is possible to do this with fancy lambda functions or std::pair in C++ too, but  I don't
    // mind writing two for loops instead ;-)
    
    weights.fill(0.0);
    int mu = eliteness; // In CMA-ES, eliteness parameter is known as "mu"
    assert(mu<costs.size());
    for (int ii=0; ii<mu; ii++)
    {
      double cur_cost = costs_sorted[ii];
      for (int jj=0; jj<costs.size(); jj++)
      {
        if (costs[jj] == cur_cost)
        {
          if (weighting_method.compare("CEM")==0)
            weights[jj] = 1.0/mu; // CEM
          else
            weights[jj] = log(mu+0.5) - log(ii+1); // CMA-ES
          break;
        }
      }
      
    }
    // For debugging
    //MatrixXd print_mat(3,costs.size());
    //print_mat.row(0) = costs_sorted;
    //print_mat.row(1) = costs;
    //print_mat.row(2) = weights;
    //cout << print_mat << endl;
  }
  else
  {
    cout << __FILE__ << ":" << __LINE__ << ":WARNING: Unknown weighting method '" << weighting_method << "'. Calling with PI-BB weighting." << endl; 
    costsToWeights(costs, "PI-BB", eliteness, weights);
    return;
  }
  
  // Relative standard deviation of total costs
  double mean = weights.mean();
  double std = sqrt((weights.array()-mean).pow(2).mean());
  double rel_std = std/mean;
  if (rel_std<1e-10)
  {
      // Special case: all costs are the same
      // Set same weights for all.
      weights.fill(1);
  }

  // Normalize weights
  weights = weights/weights.sum();

}
    
}
