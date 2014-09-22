/**
 * @file   MetaParametersGPR.cpp
 * @brief  MetaParametersGPR class source file.
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
#include "functionapproximators/MetaParametersGPR.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::MetaParametersGPR);


#include "dmpbbo_io/BoostSerializationToString.hpp"
#include "dmpbbo_io/EigenBoostSerialization.hpp"

#include <iostream>
#include <unordered_map>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/vector.hpp>



using namespace Eigen;
using namespace std;

namespace DmpBbo {

MetaParametersGPR::MetaParametersGPR(int expected_input_dim, double maximum_covariance, double length)
:
  MetaParameters(expected_input_dim),
  maximum_covariance_(maximum_covariance),
  sigmas_(VectorXd::Constant(expected_input_dim,length))
{
  assert(maximum_covariance_>0);
  assert(length>0);
}
  
MetaParametersGPR::MetaParametersGPR(int expected_input_dim, double maximum_covariance, const Eigen::VectorXd& sigmas)
:
  MetaParameters(expected_input_dim),
  maximum_covariance_(maximum_covariance),
  sigmas_(sigmas)
{
  assert(maximum_covariance_>0);
  assert(sigmas.size()==expected_input_dim);
}
  
MetaParametersGPR* MetaParametersGPR::clone(void) const
{
  return new MetaParametersGPR(getExpectedInputDim(),maximum_covariance_,sigmas_);
}

template<class Archive>
void MetaParametersGPR::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(MetaParameters);

  ar & BOOST_SERIALIZATION_NVP(maximum_covariance_);
  ar & BOOST_SERIALIZATION_NVP(sigmas_);
}

string MetaParametersGPR::toString(void) const
{
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("MetaParametersGPR");
}

}
