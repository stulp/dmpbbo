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
}

FunctionApproximatorLWR::FunctionApproximatorLWR(const ModelParametersLWR *const model_parameters) 
:
  FunctionApproximator(model_parameters)
{
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

void FunctionApproximatorLWR::train(const MatrixXd& inputs, const MatrixXd& targets)
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
  
  VectorXd min = inputs.colwise().minCoeff();
  VectorXd max = inputs.colwise().maxCoeff();
  
  MatrixXd centers, widths, activations;
  meta_parameters_lwr->getCentersAndWidths(min,max,centers,widths);
  bool normalize_activations = true; 
  bool asym_kernels = meta_parameters_lwr->asymmetric_kernels(); 
  BasisFunction::Gaussian::activations(centers,widths,inputs,activations,normalize_activations,asym_kernels);
  
  // Make the design matrix
  MatrixXd X = MatrixXd::Ones(inputs.rows(),inputs.cols()+1);
  X.leftCols(inputs.cols()) = inputs;
  
  
  int n_kernels = activations.cols();
  int n_betas = X.cols(); 
  int n_samples = X.rows(); 
  MatrixXd W;
  MatrixXd beta(n_kernels,n_betas);
  
  double epsilon = 0.000001*activations.maxCoeff();
  for (int bb=0; bb<n_kernels; bb++)
  {
    VectorXd W_vec = activations.col(bb);
    
    if (epsilon==0)
    {
      // Use all data
      
      W = W_vec.asDiagonal();
      // Compute beta
      // 1 x n_betas 
      // = inv( (n_betas x n_sam)*(n_sam x n_sam)*(n_sam*n_betas) )*( (n_betas x n_sam)*(n_sam x n_sam)*(n_sam * 1) )   
      // = inv(n_betas x n_betas)*(n_betas x 1)
      VectorXd cur_beta = (X.transpose()*W*X).inverse()*X.transpose()*W*targets;
      beta.row(bb)   =    cur_beta;
    } 
    else
    {
      // Very low weights do not contribute to the line fitting
      // Therefore, we can delete the rows in W, X and targets for which W is small
      //
      // Example with epsilon = 0.1 (a very high value!! usually it will be lower)
      //    W =       [0.001 0.01 0.5 0.98 0.46 0.01 0.001]^T
      //    X =       [0.0   0.1  0.2 0.3  0.4  0.5  0.6 ; 
      //               1.0   1.0  1.0 1.0  1.0  1.0  1.0  ]^T  (design matrix, so last column = 1)
      //    targets = [1.0   0.5  0.4 0.5  0.6  0.7  0.8  ]
      //
      // will reduce to
      //    W_sub =       [0.5 0.98 0.46 ]^T
      //    X_sub =       [0.2 0.3  0.4 ; 
      //                   1.0 1.0  1.0  ]^T  (design matrix, last column = 1)
      //    targets_sub = [0.4 0.5  0.6  ]
      // 
      // Why all this trouble? Because the submatrices will often be much smaller than the full
      // ones, so they are much faster to invert (note the .inverse() call)
      
      // Get a vector where 1 represents that W_vec >= epsilon, and 0 otherswise
      VectorXi large_enough = (W_vec.array() >= epsilon).select(VectorXi::Ones(W_vec.size()), VectorXi::Zero(W_vec.size()));

      // Number of samples in the submatrices
      int n_samples_sub = large_enough.sum();
    
      // This would be a 1-liner in Matlab... but Eigen is not good with splicing.
      VectorXd W_vec_sub(n_samples_sub);
      MatrixXd X_sub(n_samples_sub,n_betas);
      MatrixXd targets_sub(n_samples_sub,targets.cols());
      int jj=0;
      for (int ii=0; ii<n_samples; ii++)
      {
        if (large_enough[ii]==1)
        {
          W_vec_sub[jj] = W_vec[ii];
          X_sub.row(jj) = X.row(ii);
          targets_sub.row(jj) = targets.row(ii);
          jj++;
        }
      }
      
      // Do the same inversion as above, but with only a small subset of the data
      MatrixXd W_sub = W_vec_sub.asDiagonal();
      VectorXd cur_beta_sub = (X_sub.transpose()*W_sub*X_sub).inverse()*X_sub.transpose()*W_sub*targets_sub;
   
      //cout << "  n_samples=" << n_samples << endl;
      //cout << "  n_samples_sub=" << n_samples_sub << endl;
      //cout << cur_beta.transpose() << endl;
      //cout << cur_beta_sub.transpose() << endl;
      beta.row(bb)   =    cur_beta_sub;
    }
  }
  MatrixXd offsets = beta.rightCols(1);
  MatrixXd slopes = beta.leftCols(n_betas-1);
  
  setModelParameters(new ModelParametersLWR(centers,widths,slopes,offsets,asym_kernels));
  
}

void FunctionApproximatorLWR::predict(const MatrixXd& inputs, MatrixXd& output)
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

  MatrixXd lines;
  model_parameters_lwr->getLines(inputs, lines);

  // Weight the values for each line with the normalized basis function activations  
  MatrixXd activations;
  model_parameters_lwr->kernelActivations(inputs,activations);
  
  output = (lines.array()*activations.array()).rowwise().sum();
  
}

bool FunctionApproximatorLWR::saveGridData(const VectorXd& min, const VectorXd& max, const VectorXi& n_samples_per_dim, string save_directory, bool overwrite) const
{
  if (save_directory.empty())
    return true;
  
  MatrixXd inputs;
  FunctionApproximator::generateInputsGrid(min, max, n_samples_per_dim, inputs);

  const ModelParametersLWR* model_parameters_lwr = static_cast<const ModelParametersLWR*>(getModelParameters());
  
  MatrixXd lines;
  model_parameters_lwr->getLines(inputs, lines);
  
  MatrixXd unnormalized_activations;
  model_parameters_lwr->unnormalizedKernelActivations(inputs, unnormalized_activations);

  MatrixXd activations;
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
