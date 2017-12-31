/**
 * @file   ModelParametersRLS.cpp
 * @brief  ModelParametersRLS class source file.
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
#include "functionapproximators/FunctionApproximator.hpp"
#include "functionapproximators/ModelParametersRLS.hpp"
#include "functionapproximators/UnifiedModel.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::ModelParametersRLS);


#include "functionapproximators/BasisFunction.hpp"

//#include "dmpbbo_io/EigenFileIO.hpp"
#include "dmpbbo_io/BoostSerializationToString.hpp"
#include "dmpbbo_io/EigenBoostSerialization.hpp"

#include <iostream>
#include <fstream>

#include <eigen3/Eigen/Core>


using namespace std;
using namespace Eigen;

namespace DmpBbo {

ModelParametersRLS::ModelParametersRLS(const Eigen::VectorXd& slopes)
:
  slopes_(slopes), 
  offset_(0.0),
  use_offset_(false)
{
  all_values_vector_size_ += slopes_.size();
};

ModelParametersRLS::ModelParametersRLS(const Eigen::VectorXd& slopes, double offset)
:
  slopes_(slopes), 
  offset_(offset),
  use_offset_(true)
{
  all_values_vector_size_ += slopes_.size() + 1;
};

ModelParameters* ModelParametersRLS::clone(void) const {
  return new ModelParametersRLS(slopes_,offset_); 
}


void ModelParametersRLS::getLines(const Eigen::Ref<const Eigen::MatrixXd>& inputs, VectorXd& lines) const
{
  ENTERING_REAL_TIME_CRITICAL_CODE
  
  int n_time_steps = inputs.rows();
  lines.resize(n_time_steps);
  lines.noalias() = inputs*slopes_.transpose();
  if (use_offset_)
    lines.array() += offset_; // yyy Check this
  
  EXITING_REAL_TIME_CRITICAL_CODE
}


template<class Archive>
void ModelParametersRLS::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ModelParameters);

  ar & BOOST_SERIALIZATION_NVP(slopes_);
  ar & BOOST_SERIALIZATION_NVP(offset_);
  ar & BOOST_SERIALIZATION_NVP(use_offset_);
  ar & BOOST_SERIALIZATION_NVP(all_values_vector_size_);
}

string ModelParametersRLS::toString(void) const
{
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("ModelParametersRLS");
}

void ModelParametersRLS::getSelectableParameters(set<string>& selected_values_labels) const 
{
  selected_values_labels = set<string>();
  selected_values_labels.insert("slopes");
  if (use_offset_)
    selected_values_labels.insert("offsets");
}


void ModelParametersRLS::getParameterVectorMask(const std::set<std::string> selected_values_labels, VectorXi& selected_mask) const
{

  selected_mask.resize(getParameterVectorAllSize());
  selected_mask.fill(0);
  
  int offset = 0;
  
  // Slopes
  int size = slopes_.size();
  if (selected_values_labels.find("slopes")!=selected_values_labels.end())
    selected_mask.segment(offset,size).fill(1);
  offset += size;
  
  // Offsets
  if (use_offset_)
  {
    size = 1;
    if (selected_values_labels.find("offsets")!=selected_values_labels.end())
      selected_mask.segment(offset,size).fill(2);
    offset += size;
  }

  assert(offset == getParameterVectorAllSize());   
}

void ModelParametersRLS::getParameterVectorAll(VectorXd& values) const
{
  values.resize(getParameterVectorAllSize());
  values.segment(0,slopes_.size()) = slopes_;
  if (use_offset_)
    values[slopes_.size()] = offset_;
};

void ModelParametersRLS::setParameterVectorAll(const VectorXd& values) {

  if (all_values_vector_size_ != values.size())
  {
    cerr << __FILE__ << ":" << __LINE__ << ": values is of wrong size." << endl;
    return;
  }
  
  slopes_ = values.segment(0,slopes_.size());
  if (use_offset_)
    offset_ = values[slopes_.size()];

  
};

void ModelParametersRLS::setParameterVectorModifierPrivate(std::string modifier, bool new_value)
{
}

UnifiedModel* ModelParametersRLS::toUnifiedModel(void) const
{ 
  cerr << __FILE__ << ":" << __LINE__ << ":";
  cerr << "ModelParametersRLS::toUnifiedModel(void) not implemented yet, returning NULL" << endl;
  return NULL;
  //return new UnifiedModel(slopes_, offsets_); 
}

}
