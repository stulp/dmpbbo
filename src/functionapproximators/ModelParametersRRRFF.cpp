/**
 * @file   ModelParametersRRRFF.cpp
 * @brief  ModelParametersRRRFF class source file.
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
 
#include "functionapproximators/ModelParametersRRRFF.hpp"
#include "functionapproximators/BasisFunction.hpp"
#include "functionapproximators/UnifiedModel.hpp"

#include "dmpbbo_io/BoostSerializationToString.hpp"

#include <iostream>

using namespace Eigen;
using namespace std;

namespace DmpBbo {

ModelParametersRRRFF::ModelParametersRRRFF(Eigen::VectorXd weights, Eigen::MatrixXd cosines_periodes, Eigen::VectorXd cosines_phase)
:
  weights_(weights),
  cosines_periodes_(cosines_periodes),
  cosines_phase_(cosines_phase)
{

  assert(cosines_phase.size() == cosines_periodes.rows());
  assert(weights.rows() == cosines_periodes.rows());

  nb_in_dim_ = cosines_periodes.cols();
  // int nb_output_dim = weights.cols();

  
  int all_values_vector_size_ = 0;
  all_values_vector_size_ += weights_.rows() * weights_.cols();
  all_values_vector_size_ += cosines_phase_.size();
  all_values_vector_size_ += cosines_periodes_.rows() * cosines_periodes_.cols();
};

ModelParameters* ModelParametersRRRFF::clone(void) const 
{
  return new ModelParametersRRRFF(weights_, cosines_periodes_, cosines_phase_); 
}

void ModelParametersRRRFF::cosineActivations(const Eigen::Ref<const Eigen::MatrixXd>& inputs, Eigen::MatrixXd& cosine_activations) const
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

int ModelParametersRRRFF::getExpectedInputDim(void) const  
{
  return nb_in_dim_;
};

string ModelParametersRRRFF::toString(void) const 
{
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("ModelParametersRRRFF");
};


void ModelParametersRRRFF::getSelectableParameters(set<string>& selected_values_labels) const 
{
  
  selected_values_labels = set<string>();
  /*
  selected_values_labels.insert("weights");
  selected_values_labels.insert("phases");
  selected_values_labels.insert("periods");
  */
}

/*
void ModelParametersRRRFF::getParameterVectorAll(VectorXd& values) const
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

void ModelParametersRRRFF::setParameterVectorAll(const VectorXd& values)
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
*/

UnifiedModel* ModelParametersRRRFF::toUnifiedModel(void) const
{
  return new UnifiedModel(cosines_periodes_, cosines_phase_, weights_); 
}

}


