/**
 * @file   FunctionApproximatorGPR.cpp
 * @brief  FunctionApproximatorGPR class source file.
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
#include "functionapproximators/FunctionApproximatorGPR.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::FunctionApproximatorGPR);

#include "functionapproximators/ModelParametersGPR.hpp"
#include "functionapproximators/MetaParametersGPR.hpp"

#include "dmpbbo_io/EigenBoostSerialization.hpp"

#include <iostream>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/LU>

using namespace std;
using namespace Eigen;

namespace DmpBbo {

FunctionApproximatorGPR::FunctionApproximatorGPR(const MetaParametersGPR *const meta_parameters, const ModelParametersGPR *const model_parameters) 
:
  FunctionApproximator(meta_parameters,model_parameters)
{
  cerr << __FILE__ << ":" << __LINE__ << ":";
  cerr << "FunctionApproximatorGPR is still under development! No guarantees on functionality..." << endl;}

FunctionApproximatorGPR::FunctionApproximatorGPR(const ModelParametersGPR *const model_parameters) 
:
  FunctionApproximator(model_parameters)
{
  cerr << __FILE__ << ":" << __LINE__ << ":";
  cerr << "FunctionApproximatorGPR is still under development! No guarantees on functionality..." << endl;
}


FunctionApproximator* FunctionApproximatorGPR::clone(void) const {
  // All error checking and cloning is left to the FunctionApproximator constructor.
  return new FunctionApproximatorGPR(
    dynamic_cast<const MetaParametersGPR*>(getMetaParameters()),
    dynamic_cast<const ModelParametersGPR*>(getModelParameters())
    );
};



double FunctionApproximatorGPR::covarianceFunction(const VectorXd& input1, const VectorXd& input2, double maximum_covariance, double length) {
  double norm_2 = (input1-input2).squaredNorm();
  return maximum_covariance*exp(-0.5*norm_2/(length*length)); 
}



void FunctionApproximatorGPR::train(const MatrixXd& inputs, const MatrixXd& targets)
{
  if (isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorGPR::train more than once. Doing nothing." << endl;
    cerr << "   (if you really want to retrain, call reTrain function instead)" << endl;
    return;
  }
  
  assert(inputs.rows() == targets.rows());
  assert(inputs.cols()==getExpectedInputDim());

  const MetaParametersGPR* meta_parameters_gpr = 
    dynamic_cast<const MetaParametersGPR*>(getMetaParameters());
  
  double max_covar = meta_parameters_gpr->maximum_covariance();
  double length = meta_parameters_gpr->length();
  
  MatrixXd gram(inputs.rows(),inputs.rows());
  for (int ii=0; ii<inputs.rows(); ii++)
  {
    for (int jj=0; jj<inputs.rows(); jj++)
    {
      gram(ii,jj) = covarianceFunction(inputs.row(ii),inputs.row(jj),max_covar,length);
    }
  }
  setModelParameters(new ModelParametersGPR(inputs,targets,gram,max_covar,length));
  
}

void FunctionApproximatorGPR::predict(const MatrixXd& inputs, MatrixXd& outputs)
{
  if (!isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorLWPR::predict if you have not trained yet. Doing nothing." << endl;
    return;
  }

  const ModelParametersGPR* model_parameters_gpr = static_cast<const ModelParametersGPR*>(getModelParameters());
  
  MatrixXd train_inputs = model_parameters_gpr->train_inputs();
  double maximum_covariance = model_parameters_gpr->maximum_covariance();
  double length = model_parameters_gpr->length();
  VectorXd weights = model_parameters_gpr->weights();
  
  assert(inputs.cols()==getExpectedInputDim());
  unsigned int n_samples = inputs.rows();
  unsigned int n_samples_train = train_inputs.rows();
  
  outputs.resize(n_samples,1);
  
  RowVectorXd k(n_samples_train);
  for (unsigned int ii=0; ii<n_samples; ii++)
  {
    for (unsigned int jj=0; jj<n_samples_train; jj++)
      k(jj) = covarianceFunction(inputs.row(ii),train_inputs.row(jj),maximum_covariance,length);

    outputs(ii) = k*weights;
  }
  
}

void FunctionApproximatorGPR::predictVariance(const MatrixXd& inputs, MatrixXd& variances)
{
  if (!isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorLWPR::predict if you have not trained yet. Doing nothing." << endl;
    return;
  }

  const ModelParametersGPR* model_parameters_gpr = static_cast<const ModelParametersGPR*>(getModelParameters());
  
  MatrixXd train_inputs = model_parameters_gpr->train_inputs();
  double maximum_covariance = model_parameters_gpr->maximum_covariance();
  double length = model_parameters_gpr->length();
  VectorXd weights = model_parameters_gpr->weights();
  MatrixXd gram_inv = model_parameters_gpr->gram_inv();
  
  assert(inputs.cols()==getExpectedInputDim());
  unsigned int n_samples = inputs.rows();
  unsigned int n_samples_train = train_inputs.rows();
  
  variances.resize(n_samples,1);
  
  VectorXd k(n_samples_train);
  for (unsigned int ii=0; ii<n_samples; ii++)
  {
    // Covariance with the input itself
    double k_self =  FunctionApproximatorGPR::covarianceFunction(inputs.row(ii),inputs.row(ii),maximum_covariance,length);
    
    // Covariance of input with all target inputs
    for (unsigned int jj=0; jj<n_samples_train; jj++)
      k(jj) = FunctionApproximatorGPR::covarianceFunction(inputs.row(ii),train_inputs.row(jj),maximum_covariance,length);
    
    VectorXd rest = k.transpose()*gram_inv*k;
    //cout << "k=" << k.rows() << " X " << k.cols() << endl;
    //cout << "gram_inv_=" << gram_inv_.rows() << " X " << gram_inv_.cols() << endl;
    //cout << "rest=" << rest.rows() << " X " << rest.cols() << endl;
    assert(rest.rows()==1);
    assert(rest.cols()==1);
    variances(ii) = k_self - rest(0);
  }
}

template<class Archive>
void FunctionApproximatorGPR::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(FunctionApproximator);
}

}
