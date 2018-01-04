/**
 * @file   FunctionApproximatorLWR.cpp
 * @brief  FunctionApproximatorLWR class source file.
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
#include "functionapproximators/FunctionApproximatorLWR.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::FunctionApproximatorLWR);

#include "functionapproximators/ModelParametersLWR.hpp"
#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/BasisFunction.hpp"
#include "functionapproximators/leastSquares.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"
#include "dmpbbo_io/EigenBoostSerialization.hpp"

#include <iostream>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/LU>

using namespace std;
using namespace Eigen;

namespace DmpBbo {

FunctionApproximatorLWR::FunctionApproximatorLWR(const MetaParametersLWR *const meta_parameters, const ModelParametersLWR *const model_parameters) 
:
  FunctionApproximator(meta_parameters,model_parameters)
{
  if (model_parameters!=NULL)
    preallocateMemory(model_parameters->getNumberOfBasisFunctions());
}

FunctionApproximatorLWR::FunctionApproximatorLWR(const ModelParametersLWR *const model_parameters) 
:
  FunctionApproximator(model_parameters)
{
  preallocateMemory(model_parameters->getNumberOfBasisFunctions());
}

void FunctionApproximatorLWR::preallocateMemory(int n_basis_functions)
{
  lines_one_prealloc_ = MatrixXd(1,n_basis_functions);
  activations_one_prealloc_ = MatrixXd(1,n_basis_functions);
  
  lines_prealloc_ = MatrixXd(1,n_basis_functions);
  activations_prealloc_ = MatrixXd(1,n_basis_functions);
}


FunctionApproximator* FunctionApproximatorLWR::clone(void) const {
  // All error checking and cloning is left to the FunctionApproximator constructor.
  return new FunctionApproximatorLWR(
    dynamic_cast<const MetaParametersLWR*>(getMetaParameters()),
    dynamic_cast<const ModelParametersLWR*>(getModelParameters())
    );
};



///** Compute Moore-Penrose pseudo-inverse. 
// * Taken from: http://eigen.tuxfamily.org/bz/show_bug.cgi?id=257
// * \param[in]  a       The matrix to be inversed.
// * \param[out] result  The pseudo-inverse of the matrix.
// * \param[in]  epsilon Don't know, not my code ;-)
// * \return     true if pseudo-inverse possible, false otherwise
//template<typename _Matrix_Type_>
//bool pseudoInverse(const _Matrix_Type_ &a, _Matrix_Type_ &result, double
//epsilon = std::numeric_limits<typename _Matrix_Type_::Scalar>::epsilon())
//{
//  if(a.rows()<a.cols())
//      return false;
//
//  Eigen::JacobiSVD< _Matrix_Type_ > svd = a.jacobiSvd(Eigen::ComputeThinU |
//Eigen::ComputeThinV);
//
//  typename _Matrix_Type_::Scalar tolerance = epsilon * std::max(a.cols(),
//a.rows()) * svd.singularValues().array().abs().maxCoeff();
//
//  result = svd.matrixV() * _Matrix_Type_( (svd.singularValues().array().abs() >
//tolerance).select(svd.singularValues().
//      array().inverse(), 0) ).asDiagonal() * svd.matrixU().adjoint();
//      
//  return true;
//}

void FunctionApproximatorLWR::train(const Eigen::Ref<const Eigen::MatrixXd>& inputs, const Eigen::Ref<const Eigen::MatrixXd>& targets)
{
  if (isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorLWR::train more than once. Doing nothing." << endl;
    cerr << "   (if you really want to retrain, call reTrain function instead)" << endl;
    return;
  }
  
  assert(inputs.rows() == targets.rows());
  assert(inputs.cols()==getExpectedInputDim());

  const MetaParametersLWR* meta_parameters_lwr = 
    dynamic_cast<const MetaParametersLWR*>(getMetaParameters());

  // Determine the centers and widths of the basis functions, given the range of the input data
  VectorXd min = inputs.colwise().minCoeff();
  VectorXd max = inputs.colwise().maxCoeff();
  MatrixXd centers, widths;
  meta_parameters_lwr->getCentersAndWidths(min,max,centers,widths);
  bool normalize_activations = true; 
  bool asym_kernels = meta_parameters_lwr->asymmetric_kernels(); 

  // Get the activations of the basis functions 
  int n_samples = inputs.rows();
  int n_kernels = centers.rows(); 
  MatrixXd activations(n_samples,n_kernels);
  BasisFunction::Gaussian::activations(centers,widths,inputs,activations,normalize_activations,asym_kernels);
  
  // Parameters for the weighted least squares regressions
  bool use_offset = true;
  double regularization = 0.0;
  double min_weight = 0.000001*activations.maxCoeff();

  // Prepare matrices
  int n_betas = inputs.cols();
  if (use_offset)
    n_betas++;
  MatrixXd beta(n_kernels,n_betas);
  VectorXd cur_beta(n_betas);
  VectorXd weights(inputs.rows());
  
  // Perform one weighted least squares regression for each kernel
  for (int i_kernel=0; i_kernel<n_kernels; i_kernel++)
  {
    weights = activations.col(i_kernel);
    cur_beta = weightedLeastSquares(inputs,targets,weights,use_offset,regularization,min_weight);
    beta.row(i_kernel) = cur_beta;
  }
  
  MatrixXd offsets = beta.rightCols(1);
  MatrixXd slopes = beta.leftCols(n_betas-1);
  
  setModelParameters(new ModelParametersLWR(centers,widths,slopes,offsets,asym_kernels));
  
  preallocateMemory(n_kernels);
}

void FunctionApproximatorLWR::predict(const Eigen::Ref<const Eigen::MatrixXd>& inputs, MatrixXd& outputs)
{

  if (!isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorLWPR::predict if you have not trained yet. Doing nothing." << endl;
    return;
  }
  
  // The following line of code took a long time to decide on.
  // The member FunctionApproximator::model_parameters_ (which we access through
  // getModelParameters()) is of class ModelParameters, not ModelParametersLWR.
  // So within this function, we need to cast it to ModelParametersLWR in order to make predictions.
  // There are three options to do this:
  //
  // 1) use a dynamic_cast. This is really the best way to do it, but the execution of dynamic_cast
  //    can take relatively long, so we should avoid calling it in this time-critical function
  //    predict() function. (note: because it doesn't matter so much for the non time-critical
  //    train() function above, we  use the preferred dynamic_cast<MetaParametersLWR*> as we should)
  //
  // 2) move the model_parameters_ member from FunctionApproximator to FunctionApproximatorLWR, and 
  //    make it ModelParametersLWR instead of ModelParameters. This, however, will lead to lots of 
  //    code duplication, because each derived function approximator class will have to do this.
  //
  // 3) Do a static_cast. The static cast does not do checking like dynamic_cast, so we have to be
  //    really sure that getModelParameters returns a ModelParametersLWR. The only way in which this 
  //    could wrong is if someone calls setModelParameters() with a different derived class. And
  //    this is near-impossible, because setModelParameters is protected within 
  //    FunctionApproximator, and a derived class would be really dumb to set ModelParametersAAA 
  //    with setModelParameters and expect getModelParameters to return ModelParametersBBB. 
  //
  // So I decided to go with 3) because it is fast and does not lead to code duplication, 
  // and only real dumb derived classes can cause trouble ;-)
  //
  // Note: The execution time difference between 2) and 3) is negligible:  
  //   No cast    : 8.90 microseconds/prediction of 1 input sample
  //   Static cast: 8.91 microseconds/prediction of 1 input sample
  //
  // There, ~30 lines of comment for one line of code ;-) 
  //                                            (mostly for me to remember why it is like this) 
  const ModelParametersLWR* model_parameters_lwr = static_cast<const ModelParametersLWR*>(getModelParameters());
  
  bool only_one_sample = (inputs.rows()==1);
  if (only_one_sample)
  {
    ENTERING_REAL_TIME_CRITICAL_CODE

    // Only 1 sample, so real-time execution is possible. No need to allocate memory.
    model_parameters_lwr->getLines(inputs, lines_one_prealloc_);

    // Weight the values for each line with the normalized basis function activations  
    model_parameters_lwr->kernelActivations(inputs,activations_one_prealloc_);
  
    outputs = (lines_one_prealloc_.array()*activations_one_prealloc_.array()).rowwise().sum();  
    
    EXITING_REAL_TIME_CRITICAL_CODE
    
  }
  else
  {
    
    int n_time_steps = inputs.rows();
    int n_basis_functions = model_parameters_lwr->getNumberOfBasisFunctions();
    
    // The next two lines are not be real-time, as they allocate memory
    lines_prealloc_.resize(n_time_steps,n_basis_functions);
    activations_prealloc_.resize(n_time_steps,n_basis_functions);
    outputs.resize(n_time_steps,getExpectedOutputDim());
    
    model_parameters_lwr->getLines(inputs, lines_prealloc_);

    // Weight the values for each line with the normalized basis function activations  
    model_parameters_lwr->kernelActivations(inputs,activations_prealloc_);
  
    outputs = (lines_prealloc_.array()*activations_prealloc_.array()).rowwise().sum();  
    
  }
  
}

bool FunctionApproximatorLWR::saveGridData(const VectorXd& min, const VectorXd& max, const VectorXi& n_samples_per_dim, string save_directory, bool overwrite) const
{
  if (save_directory.empty())
    return true;
  
  MatrixXd inputs;
  FunctionApproximator::generateInputsGrid(min, max, n_samples_per_dim, inputs);

  const ModelParametersLWR* model_parameters_lwr = static_cast<const ModelParametersLWR*>(getModelParameters());
  
  int n_samples = inputs.rows();
  int n_basis_functions = model_parameters_lwr->getNumberOfBasisFunctions();
  
  MatrixXd lines(n_samples,n_basis_functions);
  model_parameters_lwr->getLines(inputs, lines);
  
  MatrixXd unnormalized_activations(n_samples,n_basis_functions);
  model_parameters_lwr->unnormalizedKernelActivations(inputs, unnormalized_activations);

  MatrixXd activations(n_samples,n_basis_functions);
  model_parameters_lwr->kernelActivations(inputs, activations);

  MatrixXd predictions = (lines.array()*activations.array()).rowwise().sum();
  
  saveMatrix(save_directory,"n_samples_per_dim.txt",n_samples_per_dim,overwrite);
  saveMatrix(save_directory,"inputs_grid.txt",inputs,overwrite);
  saveMatrix(save_directory,"lines_grid.txt",lines,overwrite);
  saveMatrix(save_directory,"activations_unnormalized_grid.txt",unnormalized_activations,overwrite);
  saveMatrix(save_directory,"activations_grid.txt",activations,overwrite);
  saveMatrix(save_directory,"predictions_grid.txt",predictions,overwrite);

  
  return true;
  
}

template<class Archive>
void FunctionApproximatorLWR::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(FunctionApproximator);
}

}
