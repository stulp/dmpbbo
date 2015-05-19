/**
 * @file   ModelParametersIRFRLS.cpp
 * @brief  ModelParametersIRFRLS class source file.
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
 
#include <boost/serialization/export.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include "functionapproximators/ModelParametersIRFRLS.hpp"


/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::ModelParametersIRFRLS);

#include <iostream>

#include "functionapproximators/BasisFunction.hpp"
#include "functionapproximators/UnifiedModel.hpp"
#include "dmpbbo_io/EigenBoostSerialization.hpp"
#include "dmpbbo_io/BoostSerializationToString.hpp"


using namespace Eigen;
using namespace std;

namespace DmpBbo {

ModelParametersIRFRLS::ModelParametersIRFRLS(Eigen::VectorXd weights, Eigen::MatrixXd cosines_periodes, Eigen::VectorXd cosines_phase)
:
  weights_(weights),
  cosines_periodes_(cosines_periodes),
  cosines_phase_(cosines_phase)
{

  assert(cosines_phase.size() == cosines_periodes.rows());
  assert(weights.rows() == cosines_periodes.rows());

  nb_in_dim_ = cosines_periodes.cols();
  // int nb_output_dim = weights.cols();
  
  all_values_vector_size_ = 0;
  all_values_vector_size_ += weights_.rows() * weights_.cols();
  all_values_vector_size_ += cosines_phase_.size();
  all_values_vector_size_ += cosines_periodes_.rows() * cosines_periodes_.cols();
};

ModelParameters* ModelParametersIRFRLS::clone(void) const 
{
  return new ModelParametersIRFRLS(weights_, cosines_periodes_, cosines_phase_); 
}

void ModelParametersIRFRLS::cosineActivations(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& cosine_activations) const
{
  if (caching_)
  {
    // If the cached inputs matrix has the same size as the one now requested
    //     (this also takes care of the case when inputs_cached is empty and need to be initialized)
    if ( inputs.rows()==inputs_cached_.rows() && inputs.cols()==inputs_cached_.cols() )
    {
      // And they have the same values
      if ( (inputs.array()==inputs_cached_.array()).all() )
      {
        // Then you can return the cached values
        cosine_activations = cosine_activations_cached_;
        return;
      }
    }
  }

  BasisFunction::Cosine::activations(cosines_periodes_,cosines_phase_,inputs,cosine_activations);
  
  if (caching_)
  {
    // Cache the current results now.  
    inputs_cached_ = inputs;
    cosine_activations_cached_ = cosine_activations;
  }
  
}

int ModelParametersIRFRLS::getExpectedInputDim(void) const  
{
  return nb_in_dim_;
};

template<class Archive>
void ModelParametersIRFRLS::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ModelParameters);

  ar & BOOST_SERIALIZATION_NVP(weights_);
  ar & BOOST_SERIALIZATION_NVP(cosines_periodes_);
  ar & BOOST_SERIALIZATION_NVP(cosines_phase_);
  ar & BOOST_SERIALIZATION_NVP(nb_in_dim_);
  ar & BOOST_SERIALIZATION_NVP(all_values_vector_size_);
}

string ModelParametersIRFRLS::toString(void) const 
{
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("ModelParametersIRFRLS");
};


void ModelParametersIRFRLS::getSelectableParameters(set<string>& selected_values_labels) const 
{
  selected_values_labels = set<string>();
  selected_values_labels.insert("weights");
  selected_values_labels.insert("phases");
  selected_values_labels.insert("periods");
}

void ModelParametersIRFRLS::getParameterVectorMask(const std::set<std::string> selected_values_labels, VectorXi& selected_mask) const
{
  selected_mask.resize(getParameterVectorAllSize());
  selected_mask.fill(0);
  
  int offset = 0;
  int size;
  
  size = weights_.rows() * weights_.cols();
  if (selected_values_labels.find("weights")!=selected_values_labels.end())
    selected_mask.segment(offset,size).fill(1);
  offset += size;
  
  size =  cosines_phase_.size();
  if (selected_values_labels.find("phases")!=selected_values_labels.end())
    selected_mask.segment(offset,size).fill(2);
  offset += size;
  
  size = cosines_periodes_.rows() * cosines_periodes_.cols();
  if (selected_values_labels.find("periods")!=selected_values_labels.end())
    selected_mask.segment(offset,size).fill(3);
  offset += size;
  
  assert(offset == getParameterVectorAllSize()); 
}

void ModelParametersIRFRLS::getParameterVectorAll(VectorXd& values) const
{
  values.resize(getParameterVectorAllSize());
  int offset = 0;
  
  for (int c = 0; c < weights_.cols(); c++)
  {
    values.segment(offset, weights_.rows()) = weights_.col(c);
    offset += weights_.rows();
  }

  values.segment(offset, cosines_phase_.size()) = cosines_phase_;
  offset += cosines_phase_.size();

  for (int c = 0; c < cosines_periodes_.cols(); c++)
  {
    values.segment(offset, cosines_periodes_.rows()) = cosines_periodes_.col(c);
    offset += cosines_periodes_.rows();
  }

  assert(offset == getParameterVectorAllSize()); 
  
};

void ModelParametersIRFRLS::setParameterVectorAll(const VectorXd& values)
{
  if (all_values_vector_size_ != values.size())
  {
    cerr << __FILE__ << ":" << __LINE__ << ": values is of wrong size." << endl;
    return;
  }
  
  int offset = 0;
  for (int c = 0; c < weights_.cols(); c++)
  {
    weights_.col(c) = values.segment(offset, weights_.rows());
    offset += weights_.rows();
  }
  
  cosines_phase_ = values.segment(offset, cosines_phase_.size());
  offset += cosines_phase_.size();

  for (int c = 0; c < cosines_periodes_.cols(); c++)
  {
    cosines_periodes_.col(c) = values.segment(offset, cosines_periodes_.rows());
    offset += cosines_periodes_.rows();
  }

  assert(offset == getParameterVectorAllSize());   
};

UnifiedModel* ModelParametersIRFRLS::toUnifiedModel(void) const
{
  return new UnifiedModel(cosines_periodes_, cosines_phase_, weights_); 
}

}


