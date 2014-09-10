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

#include <boost/serialization/export.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include "bbo/updaters/UpdaterMean.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::UpdaterMean);

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

/** \todo Implement other weighting schemes */  
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
  //else if (weighting_method.compare("CMA-ES")==0)
  //{
  //http://stackoverflow.com/questions/2686548/sorting-eigenvectors-by-their-eigenvalues-associated-sorting
    //std::sort(v.data(), v.data()+v.size()); 
  /*
    elseif (strcmp(weighting_method,'CEM') || strcmp(weighting_method,'CMA-ES'))
      % CEM/CMA-ES style weights: rank-based, uses defaults
      mu = eliteness; % In CMA-ES, eliteness parameter is known as "mu"
      [Ssorted indices] = sort(costs,'ascend');
      weights = zeros(size(costs));
      if (strcmp(weighting_method,'CEM'))
        weights(indices(1:mu)) = 1/mu;
      else
        for ii=1:mu
          weights(indices(ii)) = log(mu+1/2)-log(ii);
        end
      end
  */
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
