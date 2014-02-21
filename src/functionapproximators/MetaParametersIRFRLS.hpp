/**
 * @file   MetaParametersIRFRLS.hpp
 * @brief  MetaParametersIRFRLS class header file.
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

#ifndef METAPARAMETERSIRFRLS_H
#define METAPARAMETERSIRFRLS_H

#include "functionapproximators/MetaParameters.hpp"

#include <iosfwd>

namespace DmpBbo {

/** \brief Meta-parameters for the iRFRLS function approximator
 * \ingroup FunctionApproximators
 * \ingroup IRFRLS
 */
class MetaParametersIRFRLS : public MetaParameters
{
  friend class FunctionApproximatorIRFRLS;
  
public:

  /** Constructor for the algorithmic meta-parameters of the IRFRLS function approximator
   *  \param[in] expected_input_dim Expected dimensionality of the input data
   *  \param[in] number_of_basis_functions Number of basis functions
   *  \param[in] lambda Ridge regression coefficient, tradeoff between data fit and model complexity
   *  \param[in] gamma Cosines periodes distribution standard derivation
   */
	MetaParametersIRFRLS(int expected_input_dim, int number_of_basis_functions, double lambda, double gamma);
	
	MetaParametersIRFRLS* clone(void) const;
	
  std::string toString(void) const;

private:

  /** Number of basis functions */
  int number_of_basis_functions_;

  /** Ridge regression coefficient, tradeoff between data fit and model complexity */
  double lambda_;

  /** Cosines periodes distribution standard derivation */
  double gamma_;

  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
   MetaParametersIRFRLS(void) {};
   
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
BOOST_CLASS_EXPORT_KEY2(DmpBbo::MetaParametersIRFRLS, "MetaParametersIRFRLS")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::MetaParametersIRFRLS,boost::serialization::object_serializable);

#endif        //  #ifndef METAPARAMETERSIRFRLS_H

