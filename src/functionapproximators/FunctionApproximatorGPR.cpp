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

FunctionApproximatorGPR::FunctionApproximatorGPR(MetaParametersGPR *meta_parameters, ModelParametersGPR *model_parameters) 
:
  FunctionApproximator(meta_parameters,model_parameters)
{
  cerr << __FILE__ << ":" << __LINE__ << ":";
  cerr << "FunctionApproximatorGPR is still under development! No guarantees on functionality..." << endl;
}

FunctionApproximatorGPR::FunctionApproximatorGPR(ModelParametersGPR *model_parameters) 
:
  FunctionApproximator(model_parameters)
{
  cerr << __FILE__ << ":" << __LINE__ << ":";
  cerr << "FunctionApproximatorGPR is still under development! No guarantees on functionality..." << endl;
}


FunctionApproximator* FunctionApproximatorGPR::clone(void) const {

  MetaParametersGPR*  meta_params  = NULL;
  if (getMetaParameters()!=NULL)
    meta_params = dynamic_cast<MetaParametersGPR*>(getMetaParameters()->clone());

  ModelParametersGPR* model_params = NULL;
  if (getModelParameters()!=NULL)
    model_params = dynamic_cast<ModelParametersGPR*>(getModelParameters()->clone());

  if (meta_params==NULL)
    return new FunctionApproximatorGPR(model_params);
  else
    return new FunctionApproximatorGPR(meta_params,model_params);
};



/** Compute Moore-Penrose pseudo-inverse. 
 * Taken from: http://eigen.tuxfamily.org/bz/show_bug.cgi?id=257
 * \param[in]  a       The matrix to be inversed.
 * \param[out] result  The pseudo-inverse of the matrix.
 * \param[in]  epsilon Don't know, not my code ;-)
 * \return     true if pseudo-inverse possible, false otherwise
template<typename _Matrix_Type_>
bool pseudoInverse(const _Matrix_Type_ &a, _Matrix_Type_ &result, double
epsilon = std::numeric_limits<typename _Matrix_Type_::Scalar>::epsilon())
{
  if(a.rows()<a.cols())
      return false;

  Eigen::JacobiSVD< _Matrix_Type_ > svd = a.jacobiSvd(Eigen::ComputeThinU |
Eigen::ComputeThinV);

  typename _Matrix_Type_::Scalar tolerance = epsilon * std::max(a.cols(),
a.rows()) * svd.singularValues().array().abs().maxCoeff();

  result = svd.matrixV() * _Matrix_Type_( (svd.singularValues().array().abs() >
tolerance).select(svd.singularValues().
      array().inverse(), 0) ).asDiagonal() * svd.matrixU().adjoint();
      
  return true;
}
*/


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
  
  model_parameters_gpr->predictMean(inputs, outputs);
}

void FunctionApproximatorGPR::predictVariance(const MatrixXd& inputs, MatrixXd& variances)
{
  if (!isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorLWPR::predict if you have not trained yet. Doing nothing." << endl;
    return;
  }

  const ModelParametersGPR* model_parameters_gpr = static_cast<const ModelParametersGPR*>(getModelParameters());
  
  model_parameters_gpr->predictVariance(inputs, variances);
}

template<class Archive>
void FunctionApproximatorGPR::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(FunctionApproximator);
}

}
