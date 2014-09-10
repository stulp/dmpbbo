/**
 * @file   UpdaterCovarDecay.cpp
 * @brief  UpdaterCovarDecay class source file.
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
#include "bbo/updaters/UpdaterCovarDecay.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::UpdaterCovarDecay);

#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

namespace DmpBbo {
  
UpdaterCovarDecay::UpdaterCovarDecay(double eliteness, double covar_decay_factor, std::string weighting_method)
: UpdaterMean(eliteness, weighting_method), 
  covar_decay_factor_(covar_decay_factor)
{
  
  // Do some checks here
  if (covar_decay_factor_<=0 || covar_decay_factor_>=1)
  {
    double default_covar_decay_factor = 0.95;
    cout << __FILE__ << ":" << __LINE__ << ":Covar decay must be in range <0-1>, but it is " << covar_decay_factor_ << ". Setting to default: " << default_covar_decay_factor << endl;
    covar_decay_factor_ = default_covar_decay_factor;
  }

}

void UpdaterCovarDecay::updateDistribution(const DistributionGaussian& distribution, const MatrixXd& samples, const VectorXd& costs, VectorXd& weights, DistributionGaussian& distribution_new) const
{
  // Update the mean
  VectorXd mean_new;
  updateDistributionMean(distribution.mean(), samples, costs, weights, mean_new); 
  distribution_new.set_mean(mean_new);
  
  // Update the covariance matrix
  distribution_new.set_covar(covar_decay_factor_*covar_decay_factor_*distribution.covar());
  
}

}
