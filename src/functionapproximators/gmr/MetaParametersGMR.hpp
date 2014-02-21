/**
 * @file   MetaParametersGMR.hpp
 * @brief  MetaParametersGMR class header file.
 * @author Freek Stulp, Thibaut Munzer
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

#ifndef METAPARAMETERSGMR_H
#define METAPARAMETERSGMR_H

#include "functionapproximators/MetaParameters.hpp"

namespace DmpBbo {

/** \brief Meta-parameters for the GMR function approximator
 * \ingroup FunctionApproximators
 * \ingroup GMR
 */
class MetaParametersGMR : public MetaParameters
{
  friend class FunctionApproximatorGMR;
  
public:

  /** Constructor for the algorithmic meta-parameters of the GMR function approximator
   *  \param[in] expected_input_dim Expected dimensionality of the input data
   *  \param[in] number_of_gaussians Number of gaussians
   */
	MetaParametersGMR(int expected_input_dim, int number_of_gaussians);
	
	MetaParametersGMR* clone(void) const;

	std::string toString(void) const;
  
private:
  /** Number of gaussians */
  int number_of_gaussians_;

  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  MetaParametersGMR(void) {};
   
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
BOOST_CLASS_EXPORT_KEY2(DmpBbo::MetaParametersGMR, "MetaParametersGMR")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::MetaParametersGMR,boost::serialization::object_serializable);

#endif        //  #ifndef METAPARAMETERSGMR_H

