/**
 * @file   MetaParametersGPR.hpp
 * @brief  MetaParametersGPR class header file.
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
 
#ifndef METAPARAMETERSGPR_H
#define METAPARAMETERSGPR_H

#include "functionapproximators/MetaParameters.hpp"

#include <iosfwd>
#include <vector>
#include <eigen3/Eigen/Core>

namespace DmpBbo {

/** \brief Meta-parameters for the Gaussian Process Regression (GPR) function approximator
 * \ingroup FunctionApproximators
 * \ingroup GPR
 */
class MetaParametersGPR : public MetaParameters
{
  
public:
  
  /** Constructor for the algorithmic meta-parameters of the GPR function approximator.
   * \param[in] expected_input_dim Expected input dimensionality. Useful for debugging.
   *  \param[in] maximum_covariance The maximum allowable covariance of the covar function (aka sigma)
   *  \param[in] sigma    Standard deviation in the isotropic covariance function, i.e.  \$f e^{(-0.5*(\mathbf{x}-\mathbf{x}')^T * \mathbf{W} * (\mathbf{x}-\mathbf{x}'))}\$f, with \$f \mathbf{W} = \sigma^2 * \mathbf{I} \$f
   */
  MetaParametersGPR(int expected_input_dim, double maximum_covariance, double sigma);
  
  /** Constructor for the algorithmic meta-parameters of the GPR function approximator.
   * \param[in] expected_input_dim Expected input dimensionality. Useful for debugging.
   *  \param[in] maximum_covariance The maximum allowable covariance of the covar function (aka sigma)
   *  \param[in] sigmas    Standard deviation in the isotropic covariance function, i.e.  \$f e^{(-0.5*(\mathbf{x}-\mathbf{x}')^T * \mathbf{W} * (\mathbf{x}-\mathbf{x}'))}\$f, with \$f \mathbf{W} = sigmas.asDiagonal()\$f
   */
  MetaParametersGPR(int expected_input_dim, double maximum_covariance, const Eigen::VectorXd& sigmas);
		 
	MetaParametersGPR* clone(void) const;

	std::string toString(void) const;
	
	/** Return the maximum covariance of the covariance function. 
	 * \return Maximum covariance
	 */
	double maximum_covariance() const { return maximum_covariance_; }
	
	/** Return the sqrt of the diagonal of the covariance matrix in the Gaussian covariance function. 
	 * \return sqrt of the diagonal of the covariance matrix in the Gaussian covariance function
	 */
  const Eigen::VectorXd& sigmas() const { return sigmas_; }

private:
  
  double maximum_covariance_;
  
  Eigen::VectorXd sigmas_;

  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  MetaParametersGPR(void) {}; 
  
  /** Give boost serialization access to private members. */  
  friend class boost::serialization::access;
  
  /** Serialize class data members to boost archive. 
   * \param[in] ar Boost archive
   * \param[in] version Version of the class
   * See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/tutorial.html#simplecase
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version);

};

}

#include <boost/serialization/export.hpp>
/** Register this derived class. */
BOOST_CLASS_EXPORT_KEY2(DmpBbo::MetaParametersGPR, "MetaParametersGPR")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::MetaParametersGPR,boost::serialization::object_serializable);

#endif        //  #ifndef METAPARAMETERSGPR_H

