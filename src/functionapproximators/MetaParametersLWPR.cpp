/**
 * @file   MetaParametersLWPR.cpp
 * @brief  MetaParametersLWPR class source file.
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
#include "functionapproximators/MetaParametersLWPR.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::MetaParametersLWPR);

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/vector.hpp>

#include <iostream>

#include "dmpbbo_io/EigenBoostSerialization.hpp"
#include "dmpbbo_io/BoostSerializationToString.hpp"

using namespace Eigen;
using namespace std;

namespace DmpBbo {

MetaParametersLWPR::MetaParametersLWPR(
    int expected_input_dim, 
    Eigen::VectorXd init_D, double w_gen, double w_prune,
    bool update_D, double init_alpha, double penalty, bool diag_only,
    bool use_meta, double meta_rate, std::string kernel_name
  )
:
  MetaParameters(expected_input_dim),
  init_D_(init_D), w_gen_(w_gen), w_prune_(w_prune),
  update_D_(update_D), init_alpha_(init_alpha), penalty_(penalty), diag_only_(diag_only),
  use_meta_(use_meta), meta_rate_(meta_rate), kernel_name_(kernel_name)
{
  assert(init_D_.size()==expected_input_dim);
  assert(w_gen_>0.0 && w_gen_<1.0);
  assert(w_prune_>0.0 && w_prune_<1.0);
  assert(w_gen_<w_prune_);
}

MetaParametersLWPR* MetaParametersLWPR::clone(void) const
{
  return new MetaParametersLWPR(
    getExpectedInputDim(), 
    init_D_, w_gen_, w_prune_,
    update_D_, init_alpha_, penalty_, diag_only_,
    use_meta_, meta_rate_, kernel_name_
  );  
}

string MetaParametersLWPR::toString(void) const {
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("MetaParametersLWPR");
}


template<class Archive>
void MetaParametersLWPR::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(MetaParameters);

  ar & BOOST_SERIALIZATION_NVP(init_D_);
  ar & BOOST_SERIALIZATION_NVP(w_gen_);
  ar & BOOST_SERIALIZATION_NVP(w_prune_);
  ar & BOOST_SERIALIZATION_NVP(update_D_);
  ar & BOOST_SERIALIZATION_NVP(init_alpha_);
  ar & BOOST_SERIALIZATION_NVP(penalty_);
  ar & BOOST_SERIALIZATION_NVP(diag_only_);
  ar & BOOST_SERIALIZATION_NVP(use_meta_);
  ar & BOOST_SERIALIZATION_NVP(meta_rate_);
  ar & BOOST_SERIALIZATION_NVP(kernel_name_);

}

}

