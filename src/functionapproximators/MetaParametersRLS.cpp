/**
 * @file   MetaParametersRLS.cpp
 * @brief  MetaParametersRLS class source file.
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
#include "functionapproximators/MetaParametersRLS.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::MetaParametersRLS);


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

MetaParametersRLS::MetaParametersRLS(int expected_input_dim, double regularization, bool use_offset)
:
  MetaParameters(expected_input_dim),
  regularization_(regularization),
  use_offset_(use_offset)
{
  assert(regularization>=0.0);
}

MetaParametersRLS* MetaParametersRLS::clone(void) const
{
  MetaParametersRLS* cloned = new MetaParametersRLS(getExpectedInputDim(),regularization_);
  return cloned;
}

template<class Archive>
void MetaParametersRLS::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(MetaParameters);

  ar & BOOST_SERIALIZATION_NVP(regularization_);
  ar & BOOST_SERIALIZATION_NVP(use_offset_);
}

string MetaParametersRLS::toString(void) const
{
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("MetaParametersRLS");
}

}
