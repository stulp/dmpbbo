/**
 * @file   UpdaterCovarDecay.hpp
 * @brief  UpdaterCovarDecay class header file.
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
 
#ifndef UPDATERCOVARDECAY_H
#define UPDATERCOVARDECAY_H

#include <string>
#include <eigen3/Eigen/Core>

#include "bbo/updaters/UpdaterMean.hpp"

namespace DmpBbo {

/** Updater that updates the mean and decreases the size of the covariance matrix over time.
 */
class UpdaterCovarDecay : public UpdaterMean
{
private:
  double covar_decay_factor_;

public:
  /** Constructor
   * \param[in] eliteness The eliteness parameter ('mu' in CMA-ES, 'h' in PI^2)
   * \param[in] covar_decay_factor The covar matrix shrinks at each update with C^new = covar_decay_factor^2 * C. It should be in the range <0-1] 
   * \param[in] weighting_method ('PI-BB' = PI^2 style weighting)
   */
  UpdaterCovarDecay(double eliteness, double covar_decay_factor=0.95, std::string weighting_method="PI-BB");
  
  void updateDistribution(const DistributionGaussian& distribution, const Eigen::MatrixXd& samples, const Eigen::VectorXd& costs, Eigen::VectorXd& weights, DistributionGaussian& distribution_new) const;
  
};

}

#endif
