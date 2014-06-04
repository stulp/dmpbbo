/**
 * @file   ModelParametersGPR.cpp
 * @brief  ModelParametersGPR class source file.
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
#include "functionapproximators/ModelParametersGPR.hpp"
#include "functionapproximators/FunctionApproximatorGPR.hpp"

BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::ModelParametersGPR);

#include "dmpbbo_io/EigenFileIO.hpp"
#include "dmpbbo_io/BoostSerializationToString.hpp"
#include "dmpbbo_io/EigenBoostSerialization.hpp"

#include <iostream>
#include <fstream>

#include <eigen3/Eigen/Core>


using namespace std;
using namespace Eigen;

namespace DmpBbo {

ModelParametersGPR::ModelParametersGPR(MatrixXd train_inputs, VectorXd gram_inv_targets, double maximum_covariance, double length)
:
  train_inputs_(train_inputs),
  gram_inv_targets_(gram_inv_targets),
  maximum_covariance_(maximum_covariance),
  length_(length)
{
  assert(maximum_covariance_>0);
  assert(length_>0);
}

ModelParameters* ModelParametersGPR::clone(void) const {
  return new ModelParametersGPR(train_inputs_,gram_inv_targets_,maximum_covariance_,length_); 
}

void ModelParametersGPR::predictMean(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs) const
{
  assert(inputs.cols()==getExpectedInputDim());
  unsigned int n_samples = inputs.rows();
  unsigned int n_samples_train = train_inputs_.rows();
  
  outputs.resize(n_samples,1);
  
  RowVectorXd K(n_samples_train);
  for (unsigned int ii=0; ii<n_samples; ii++)
  {
    for (unsigned int jj=0; jj<n_samples_train; jj++)
      K(jj) = FunctionApproximatorGPR::covarianceFunction(inputs.row(ii),train_inputs_.row(jj),maximum_covariance_,length_);

    outputs(ii) = K*gram_inv_targets_;
  }
  
}

template<class Archive>
void ModelParametersGPR::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ModelParameters);

  ar & BOOST_SERIALIZATION_NVP(train_inputs_);
  ar & BOOST_SERIALIZATION_NVP(gram_inv_targets_);
  ar & BOOST_SERIALIZATION_NVP(maximum_covariance_);            
  ar & BOOST_SERIALIZATION_NVP(length_);     
                                    
}

string ModelParametersGPR::toString(void) const
{
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("ModelParametersGPR");
}

void ModelParametersGPR::getSelectableParameters(set<string>& selected_values_labels) const 
{
  selected_values_labels = set<string>();
}


void ModelParametersGPR::getParameterVectorMask(const std::set<std::string> selected_values_labels, VectorXi& selected_mask) const
{
  selected_mask.resize(0);
}

void ModelParametersGPR::getParameterVectorAll(VectorXd& values) const
{
  values.resize(getParameterVectorAllSize());
  values.resize(0);
};

void ModelParametersGPR::setParameterVectorAll(const VectorXd& values) {
};

bool ModelParametersGPR::saveGridData(const VectorXd& min, const VectorXd& max, const VectorXi& n_samples_per_dim, string save_directory, bool overwrite) const
{
  if (save_directory.empty())
    return true;
  
  //cout << "        Saving GPR model grid data to: " << save_directory << "." << endl;
  
  int n_dims = min.size();
  assert(n_dims==max.size());
  assert(n_dims==n_samples_per_dim.size());
  
  cerr << __FILE__ << ":" << __LINE__ << ":";
  cerr << "Implement this" << endl;

  
  return true;
  
}


}
