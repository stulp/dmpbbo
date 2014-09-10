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
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/LU>


using namespace std;
using namespace Eigen;

namespace DmpBbo {

ModelParametersGPR::ModelParametersGPR(Eigen::MatrixXd train_inputs, Eigen::VectorXd train_targets, Eigen::MatrixXd gram, double maximum_covariance, double length)
:
  train_inputs_(train_inputs),
  train_targets_(train_targets),
  gram_(gram),
  maximum_covariance_(maximum_covariance),
  length_(length)
{
  assert(gram_.rows()==gram_.cols());
  assert(train_inputs_.rows()==gram_.rows());
  assert(train_inputs_.rows()==train_targets_.rows());
  
  gram_inv_ = gram_.inverse();
  gram_inv_targets_ = gram_inv_ * train_targets_.col(0);

  assert(maximum_covariance_>0);
  assert(length_>0);
}

ModelParameters* ModelParametersGPR::clone(void) const {
  return new ModelParametersGPR(train_inputs_,train_targets_,gram_,maximum_covariance_,length_); 
}

void ModelParametersGPR::predictMean(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs) const
{
  assert(inputs.cols()==getExpectedInputDim());
  unsigned int n_samples = inputs.rows();
  unsigned int n_samples_train = train_inputs_.rows();
  
  outputs.resize(n_samples,1);
  
  RowVectorXd k(n_samples_train);
  for (unsigned int ii=0; ii<n_samples; ii++)
  {
    for (unsigned int jj=0; jj<n_samples_train; jj++)
      k(jj) = FunctionApproximatorGPR::covarianceFunction(inputs.row(ii),train_inputs_.row(jj),maximum_covariance_,length_);

    outputs(ii) = k*gram_inv_targets_;
  }
  
}


void ModelParametersGPR::predictVariance(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& variance) const
{
  assert(inputs.cols()==getExpectedInputDim());
  unsigned int n_samples = inputs.rows();
  unsigned int n_samples_train = train_inputs_.rows();
  
  variance.resize(n_samples,1);
  
  VectorXd k(n_samples_train);
  for (unsigned int ii=0; ii<n_samples; ii++)
  {
    // Covariance with the input itself
    double k_self =  FunctionApproximatorGPR::covarianceFunction(inputs.row(ii),inputs.row(ii),maximum_covariance_,length_);
    
    // Covariance of input with all target inputs
    for (unsigned int jj=0; jj<n_samples_train; jj++)
      k(jj) = FunctionApproximatorGPR::covarianceFunction(inputs.row(ii),train_inputs_.row(jj),maximum_covariance_,length_);
    
    VectorXd rest = k.transpose()*gram_inv_*k;
    //cout << "k=" << k.rows() << " X " << k.cols() << endl;
    //cout << "gram_inv_=" << gram_inv_.rows() << " X " << gram_inv_.cols() << endl;
    //cout << "rest=" << rest.rows() << " X " << rest.cols() << endl;
    assert(rest.rows()==1);
    assert(rest.cols()==1);
    variance(ii) = k_self - rest(0);
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
  
  MatrixXd inputs;
  FunctionApproximator::generateInputsGrid(min, max, n_samples_per_dim, inputs);

  cerr << __FILE__ << ":" << __LINE__ << ":";
  cerr << "Implement this" << endl;

  /*
  MatrixXd lines;
  getLines(inputs, lines);
  
  MatrixXd activations;
  kernelActivations(inputs, activations);
    
  MatrixXd normalized_activations;
  normalizedKernelActivations(inputs, normalized_activations);
    
  saveMatrix(save_directory,"n_samples_per_dim.txt",n_samples_per_dim,overwrite);
  saveMatrix(save_directory,"inputs_grid.txt",inputs,overwrite);
  saveMatrix(save_directory,"lines.txt",lines,overwrite);
  saveMatrix(save_directory,"activations.txt",activations,overwrite);
  saveMatrix(save_directory,"activations_normalized.txt",normalized_activations,overwrite);
  */
  
  return true;
  
}


}
