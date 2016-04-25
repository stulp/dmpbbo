/**
 * @file   Updater.hpp
 * @brief  Updater class header file.
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
 

#ifndef UPDATER_H
#define UPDATER_H

#include <vector>
#include <eigen3/Eigen/Core>

#include "bbo/DistributionGaussian.hpp"

#include <boost/serialization/access.hpp>

namespace DmpBbo {

/** Interface for the distribution update step in evolution strategies.
 *
 * Evolution strategies implement the following loop:
\code
// distribution has been initialized above
for (int i_update=1; i_update<=n_updates; i_update++)
  // Sample from distribution
  samples = distribution->generateSamples(n_samples_per_update);
  // Perform rollouts for the samples and compute costs
  costs = cost_function->evaluate(samples);
  // Update parameters
  distribution = updater->updateDistribution(distribution, samples, costs);
}
\endcode
 *
 * The last step (updating the distribution) is implemented by classes inheriting from this Updater
 * interface.
 *
 * \todo Implement << and virtual toString with boost serialization
 */
class Updater
{
public:
  

  /** Update a distribution given the samples and costs of an epoch.
   * \param[in] distribution Current distribution
   * \param[in] samples The samples in the epoch (size: n_samples X n_dims)
   * \param[in] costs Costs of the samples (size: n_samples x 1)
   * \param[out] distribution_new Updated distribution
   */
  inline void updateDistribution(const DistributionGaussian& distribution, const Eigen::MatrixXd& samples, const Eigen::VectorXd& costs, DistributionGaussian& distribution_new) const {
    Eigen::VectorXd weights;
    updateDistribution(distribution, samples, costs, weights, distribution_new);     
  }
  
  /** Update a distribution given the samples and costs of an epoch.
   * \param[in] distribution Current distribution
   * \param[in] samples The samples in the epoch (size: n_samples X n_dims)
   * \param[in] costs Costs of the samples (size: n_samples x 1)
   * \param[out] weights The weights computed from the costs
   * \param[out] distribution_new Updated distribution
   */
  virtual void updateDistribution(const DistributionGaussian& distribution, const Eigen::MatrixXd& samples, const Eigen::VectorXd& costs, Eigen::VectorXd& weights, DistributionGaussian& distribution_new) const = 0;
  
private:
  /** Give boost serialization access to private members. */  
  friend class boost::serialization::access;
  
  /** Serialize class data members to boost archive. 
   * \param[in] ar Boost archive
   * \param[in] version Version of the class
   * See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/tutorial.html#simplecase
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    // No members to serialize.
  }
  
};

} // namespace DmpBbo

#include <boost/serialization/assume_abstract.hpp>
/** Don't add version information to archives. */
BOOST_SERIALIZATION_ASSUME_ABSTRACT(DmpBbo::Updater);
 
#include <boost/serialization/level.hpp>
/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::Updater,boost::serialization::object_serializable);

#endif
