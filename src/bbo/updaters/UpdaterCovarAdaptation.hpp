/**
 * @file   UpdaterCovarAdaptation.hpp
 * @brief  UpdaterCovarAdaptation class header file.
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

#ifndef UPDATERCOVARADAPTATION_H
#define UPDATERCOVARADAPTATION_H

#include <string>
#include <eigen3/Eigen/Core>

#include "bbo/updaters/UpdaterMean.hpp"

namespace DmpBbo {

/** Updater that updates the mean and also implements Covariance Matrix Adaptation.
 * The update rule is as in <A HREF="http://en.wikipedia.org/wiki/CMA-ES">CMA-ES</A>, except that this version does not use the evolution paths.
 *
 */
class UpdaterCovarAdaptation : public UpdaterMean
{
public:
  
  /** Constructor
   * \param[in] eliteness The eliteness parameter ('mu' in CMA-ES, 'h' in PI^2)
   * \param[in] weighting_method ('PI-BB' = PI^2 style weighting)
   * \param[in] base_level Small covariance matrix that is added after each update to avoid premature convergence
   * \param[in] diag_only Update only the diagonal of the covariance matrix (true) or the full matrix (false)
   * \param[in] learning_rate Low pass filter on the covariance updates. In range [0.0-1.0] with 0.0 = no updating, 1.0  = complete update by ignoring previous covar matrix. 
   * \param[in] relative_lower_bound Enforces a lower bound on the eigen values of the covariance matrix, relative to the largest eigenvalue. E.g. if relative_lower_bound=0.1, than no eigenvalue may be smaller than 10% of the largest eigenvalue. 
   */
  UpdaterCovarAdaptation(double eliteness, std::string weighting_method="PI-BB", const Eigen::VectorXd& base_level=Eigen::VectorXd::Zero(0), bool diag_only=true, double learning_rate=1.0, double relative_lower_bound=0.0);
  
  void updateDistribution(const DistributionGaussian& distribution, const Eigen::MatrixXd& samples, const Eigen::VectorXd& costs, Eigen::VectorXd& weights, DistributionGaussian& distribution_new) const;
  
private:
  bool diag_only_;
  double learning_rate_;
  Eigen::VectorXd base_level_diagonal_;
  double relative_lower_bound_;
 
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. See \ref sec_boost_serialization_ugliness
   */
  UpdaterCovarAdaptation(void) {};
  
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
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(UpdaterMean);
    
    ar & BOOST_SERIALIZATION_NVP(diag_only_);
    ar & BOOST_SERIALIZATION_NVP(learning_rate_);
    ar & BOOST_SERIALIZATION_NVP(base_level_diagonal_);
  }

};

}

#include <boost/serialization/export.hpp>

/** Register this derived class. */
BOOST_CLASS_EXPORT_KEY2(DmpBbo::UpdaterCovarAdaptation, "UpdaterCovarAdaptation")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::UpdaterCovarAdaptation,boost::serialization::object_serializable)

#endif
