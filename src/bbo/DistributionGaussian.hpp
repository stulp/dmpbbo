/**
 * @file   DistributionGaussian.hpp
 * @brief  DistributionGaussian class header file.
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

#ifndef DISTRIBUTIONGAUSSIAN_H
#define DISTRIBUTIONGAUSSIAN_H   

#include "dmpbbo_io/EigenBoostSerialization.hpp"

#include <vector>
#include <sstream>
#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Cholesky>

#include <boost/random.hpp>

namespace DmpBbo {

/** \brief A class for representing a Gaussian distribution.
 *
 * This is mainly a wrapper around boost functionality
 * The reason to make the wrapper is to provide functionality for serialization/deserialization.
 */
class DistributionGaussian
{
public:
  /** Construct the Gaussian distribution with a mean and covariance matrix.
   *  \param[in] mean Mean of the distribution
   *  \param[in] covar Covariance matrix of the distribution
   */
  DistributionGaussian(const Eigen::VectorXd& mean, const Eigen::MatrixXd& covar);
  
  /** Generate samples from the distribution.
   *  \param[in] n_samples Number of samples to sample
   *  \param[in] samples the samples themselves (size n_samples X dim(mean)
   */
  void generateSamples(int n_samples, Eigen::MatrixXd& samples) const;
  
  /** Get the largest eigenvalue of the covariance matrix of this distribution.
   *  \return largest eigenvalue of the covariance matrix of this distribution.
   */
  double maxEigenValue(void) const;

  /** Make a deep copy of the object.
   * \return A deep copy of the object.
   */
  DistributionGaussian* clone(void) const;
  
  /**
   * Accessor get function for the mean.
   * \return The mean of the distribution
   */
  const Eigen::VectorXd& mean(void) const   { return mean_;   }
  
  /**
   * Accessor get function for the covariance matrix.
   * \return The covariance matrix of the distribution
   */
  const Eigen::MatrixXd& covar(void) const { return covar_; }
  
  /**
   * Accessor set function for the mean.
   * \param[in] mean The new mean of the distribution
   */
  void set_mean(const Eigen::VectorXd& mean);
  
  /**
   * Accessor set function for the covar.
   * \param[in] covar The new covariance matrix of the distribution
   */
  void set_covar(const Eigen::MatrixXd& covar);

  /** Print to output stream. 
   *
   *  \param[in] output  Output stream to which to write to
   *  \param[in] distribution Distribution to write
   *  \return    Output stream
   */ 
  friend std::ostream& operator<<(std::ostream& output, const DistributionGaussian& distribution);
  
private:
  /** Boost's random number generator. Shared by all object instances. */
  static boost::mt19937 rng;
  
  /** Generator for samples from a univariate unit normal distribution. Shared by all object instances. */
  static boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > *normal_distribution_unit;

  /** Mean of the distribution */
  Eigen::VectorXd mean_;
  /** Covariance matrix of the distribution */
  Eigen::MatrixXd covar_;  
  /** Cholesky decomposition of the covariance matrix of the distribution. This cached variable makes it easier to generate samples. */
  mutable Eigen::MatrixXd covar_decomposed_; 
  
  /** Maximum eigen value of the covariance matrix of the distribution. This cached variable avoid recomputing it every time maxEigenValue is called. */
  mutable double max_eigen_value_ = -1.0;
  
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
    ar & BOOST_SERIALIZATION_NVP(mean_);
    ar & BOOST_SERIALIZATION_NVP(covar_);
  }

};

}

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::DistributionGaussian,boost::serialization::object_serializable);



#endif
