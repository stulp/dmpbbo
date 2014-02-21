/**
 * @file   UpdaterMean.hpp
 * @brief  UpdaterMean class header file.
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

#ifndef UPDATERMEAN_H
#define UPDATERMEAN_H   

#include <string>
#include <eigen3/Eigen/Core>

#include "bbo/Updater.hpp"

namespace DmpBbo {

/** Updater that updates the mean (but not the covariance matrix) of the parameter distribution.
 */
class UpdaterMean : public Updater
{
public:
  
  /** Constructor
   * \param[in] eliteness The eliteness parameter ('mu' in CMA-ES, 'h' in PI^2)
   * \param[in] weighting_method ('PI-BB' = PI^2 style weighting)
   */
  UpdaterMean(double eliteness, std::string weighting_method="PI-BB");
  
  virtual void updateDistribution(const DistributionGaussian& distribution, const Eigen::MatrixXd& samples, const Eigen::VectorXd& costs, Eigen::VectorXd& weights, DistributionGaussian& distribution_new) const;
  
  /** Update the distribution mean
   * \param[in] mean Current mean
   * \param[in] samples The samples in the epoch (size: n_samples X n_dims)
   * \param[in] costs Costs of the samples (size: n_samples x 1)
   * \param[out] weights The weights computed from the costs
   * \param[out] mean_new Updated mean
   */ 
  void updateDistributionMean(const Eigen::VectorXd& mean, const Eigen::MatrixXd& samples, const Eigen::VectorXd& costs, Eigen::VectorXd& weights, Eigen::VectorXd& mean_new) const;

  /** Convert costs to weights, given the weighting method.
   *  The weights should sum to 1, and higher costs should lead to lower weights.
   * \param[in] costs Costs of the samples (size: n_samples x 1)
   * \param[in] weighting_method ('PI-BB' = PI^2 style weighting)
   * \param[in] eliteness The eliteness parameter ('mu' in CMA-ES, 'h' in PI^2)
   * \param[out] weights The weights computed from the costs
   */
  void costsToWeights(const Eigen::VectorXd& costs, std::string weighting_method, double eliteness, Eigen::VectorXd& weights) const;
  
protected:
  /** Eliteness parameters ('mu' in CMA-ES, 'h' in PI^2) */
  double eliteness_;
  /** Weighting method */
  std::string weighting_method_;

protected:
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. See \ref sec_boost_serialization_ugliness
   */
  UpdaterMean(void) {};
  
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
    // serialize base class information
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Updater);
    
    ar & BOOST_SERIALIZATION_NVP(eliteness_);
    ar & BOOST_SERIALIZATION_NVP(weighting_method_);
  }

};

}

#include <boost/serialization/export.hpp>

/** Register this derived class. */
BOOST_CLASS_EXPORT_KEY2(DmpBbo::UpdaterMean, "UpdaterMean")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::UpdaterMean,boost::serialization::object_serializable)

#endif
