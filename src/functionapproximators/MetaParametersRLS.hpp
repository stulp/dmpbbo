/**
 * @file   MetaParametersRLS.hpp
 * @brief  MetaParametersRLS class header file.
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
 
#ifndef METAPARAMETERSRLS_H
#define METAPARAMETERSRLS_H

#include "functionapproximators/MetaParameters.hpp"

#include <iosfwd>
#include <vector>
#include <eigen3/Eigen/Core>

namespace DmpBbo {

/** \brief Meta-parameters for the (Regularized) Least Squares Regression (RLS) function approximator
 * \ingroup FunctionApproximators
 * \ingroup RLS
 */
class MetaParametersRLS : public MetaParameters
{
  
public:
  
  /** Constructor for the algorithmic meta-parameters of the RLS function approximator.
   *  \param[in] expected_input_dim         The dimensionality of the data this function approximator expects. Although this information is already contained in the 'centers_per_dim' argument, we ask the user to pass it explicitly so that various checks on the arguments may be conducted.
   *  \param[in] regularization Regularization term for least-squares regression
   *  \param[in] use_offset Should the line have an offset, i.e. the b term in f(x) = ax + b
   */
   MetaParametersRLS(int expected_input_dim, double regularization=0.0, bool use_offset=true);
		 
	
	/** Accessor function for regularization.
	 * \return Regularization term.
	 */
  bool regularization(void) const
  {
    return regularization_;
  }
  
	/** Accessor function for use_offset.
	 * \return Regularization term.
	 */
  bool use_offset(void) const
  {
    return use_offset_;
  }

	MetaParametersRLS* clone(void) const;

	std::string toString(void) const;

private:
  bool regularization_; // should be const
  bool use_offset_;
  
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  MetaParametersRLS(void) {}; 
  
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
BOOST_CLASS_EXPORT_KEY2(DmpBbo::MetaParametersRLS, "MetaParametersRLS")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::MetaParametersRLS,boost::serialization::object_serializable);

#endif        //  #ifndef METAPARAMETERSRLS_H

