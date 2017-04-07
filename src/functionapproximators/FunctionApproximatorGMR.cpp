/**
 * @file   FunctionApproximatorGMR.cpp
 * @brief  FunctionApproximator class source file.
 * @author Thibaut Munzer, Freek Stulp
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
#include "functionapproximators/FunctionApproximatorGMR.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::FunctionApproximatorGMR);

#include <iostream> 
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Cholesky>
#include <ctime>
#include <cstdlib>


#include "functionapproximators/ModelParametersGMR.hpp"
#include "functionapproximators/MetaParametersGMR.hpp"
#include "dmpbbo_io/EigenBoostSerialization.hpp"


using namespace std;
using namespace Eigen;

namespace DmpBbo {

FunctionApproximatorGMR::FunctionApproximatorGMR(const MetaParametersGMR *const meta_parameters, const ModelParametersGMR *const model_parameters) 
:
  FunctionApproximator(meta_parameters,model_parameters)
{
  // TODO : find a more appropriate place for rand initialization
  //srand(unsigned(time(0)));
  if (model_parameters!=NULL)
    preallocateMatrices(
      model_parameters->getNumberOfGaussians(),
      model_parameters->getExpectedInputDim(),
      model_parameters->getExpectedOutputDim()
    );
}

FunctionApproximatorGMR::FunctionApproximatorGMR(const ModelParametersGMR *const model_parameters) 
:
  FunctionApproximator(model_parameters)
{
    preallocateMatrices(
      model_parameters->getNumberOfGaussians(),
      model_parameters->getExpectedInputDim(),
      model_parameters->getExpectedOutputDim()
    );
}

void FunctionApproximatorGMR::preallocateMatrices(int n_gaussians, int n_input_dims, int n_output_dims)
{
  probabilities_prealloc_ = VectorXd::Zero(n_gaussians);
  probabilities_dot_prealloc_ = VectorXd::Zero(n_gaussians);
  diff_prealloc_ = VectorXd::Zero(n_input_dims);
  covar_times_diff_prealloc_ = VectorXd::Zero(n_input_dims);  
  mean_output_prealloc_ = VectorXd::Zero(n_output_dims);  
  
  covar_input_times_output_ = MatrixXd::Zero(n_input_dims,n_output_dims);
  covar_output_times_input_ = MatrixXd::Zero(n_output_dims,n_input_dims);
  covar_output_prealloc_ = MatrixXd::Zero(n_output_dims,n_output_dims);
  
  empty_prealloc_ = MatrixXd::Zero(0,0);

}


FunctionApproximator* FunctionApproximatorGMR::clone(void) const {
  // All error checking and cloning is left to the FunctionApproximator constructor.
  return new FunctionApproximatorGMR(
    dynamic_cast<const MetaParametersGMR*>(getMetaParameters()),
    dynamic_cast<const ModelParametersGMR*>(getModelParameters())
    );
};


void FunctionApproximatorGMR::train(const MatrixXd& inputs, const MatrixXd& targets)
{
  if (isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorGMR::train more than once. Doing nothing." << endl;
    cerr << "   (if you really want to retrain, call reTrain function instead)" << endl;
    return;
  }
  
  assert(inputs.rows() == targets.rows()); // Must have same number of examples
  assert(inputs.cols() == getExpectedInputDim());

  const MetaParametersGMR* meta_parameters_GMR = 
    static_cast<const MetaParametersGMR*>(getMetaParameters());

  const ModelParametersGMR* model_parameters_GMR =
    static_cast<const ModelParametersGMR*>(getModelParameters());

  int n_gaussians;
  if(meta_parameters_GMR!=NULL)
      n_gaussians = meta_parameters_GMR->number_of_gaussians_;
  else if(model_parameters_GMR!=NULL)
      n_gaussians = model_parameters_GMR->priors_.size();
  else
      cerr << "FunctionApproximatorGMR::train Something wrong happened, both ModelParameters and MetaParameters are not initialized." << endl;

  int n_dims_in = inputs.cols();
  int n_dims_out = targets.cols();
  int n_dims_gmm = n_dims_in + n_dims_out;
  
  // Initialize the means, priors and covars
  std::vector<VectorXd> means(n_gaussians);
  std::vector<MatrixXd> covars(n_gaussians);
  std::vector<double> priors(n_gaussians);
  int n_observations = 0;

  for (int i = 0; i < n_gaussians; i++)
  {
    means[i] = VectorXd(n_dims_gmm);
    priors[i] = 0.0;
    covars[i] = MatrixXd(n_dims_gmm, n_dims_gmm);
  }
  
  // Put the input/output data in one big matrix
  MatrixXd data = MatrixXd(inputs.rows(), n_dims_gmm);
  data << inputs, targets;
  n_observations = data.rows();

  // Initialization
  if (inputs.cols() == 1)
    firstDimSlicingInit(data, means, priors, covars);
  else
    kMeansInit(data, means, priors, covars);
  
  // Expectation-Maximization
  expectationMaximization(data, means, priors, covars);

  // Extract the different input/output components from the means/covars which contain both
  std::vector<Eigen::VectorXd> means_x(n_gaussians);
  std::vector<Eigen::VectorXd> means_y(n_gaussians);
  std::vector<Eigen::MatrixXd> covars_x(n_gaussians);
  std::vector<Eigen::MatrixXd> covars_y(n_gaussians);
  std::vector<Eigen::MatrixXd> covars_y_x(n_gaussians);
  for (int i_gau = 0; i_gau < n_gaussians; i_gau++)
  {
    means_x[i_gau]    = means[i_gau].segment(0, n_dims_in);
    means_y[i_gau]    = means[i_gau].segment(n_dims_in, n_dims_out);

    covars_x[i_gau]   = covars[i_gau].block(0, 0, n_dims_in, n_dims_in);
    covars_y[i_gau]   = covars[i_gau].block(n_dims_in, n_dims_in, n_dims_out, n_dims_out);
    covars_y_x[i_gau] = covars[i_gau].block(n_dims_in, 0, n_dims_out, n_dims_in);
  }

  setModelParameters(new ModelParametersGMR(n_observations, priors, means_x, means_y, covars_x, covars_y, covars_y_x));

  // After training, we know the sizes of the matrices that should be cached
  preallocateMatrices(n_gaussians,n_dims_in,n_dims_out);
  
  // std::vector<VectorXd> centers;
  // std::vector<MatrixXd> slopes;
  // std::vector<VectorXd> biases;
  // std::vector<MatrixXd> inverseCovarsL;

  // // int n_dims_in = inputs.cols();
  // // int n_dims_out = targets.cols();

  // for (int i_gau = 0; i_gau < n_gaussians; i_gau++)
  // {
  //   centers.push_back(VectorXd(means[i_gau].segment(0, n_dims_in)));

  //   slopes.push_back(MatrixXd(covars[i_gau].block(n_dims_in, 0, n_dims_out, n_dims_in) * covars[i_gau].block(0, 0, n_dims_in, n_dims_in).inverse()));
    
  //   biases.push_back(VectorXd(means[i_gau].segment(n_dims_in, n_dims_out) -
  //     slopes[i_gau]*means[i_gau].segment(0, n_dims_in)));

  //   MatrixXd L = covars[i_gau].block(0, 0, n_dims_in, n_dims_in).inverse().llt().matrixL();
  //   inverseCovarsL.push_back(MatrixXd(L));
  // }

  // setModelParameters(new ModelParametersGMR(centers, priors, slopes, biases, inverseCovarsL));

  //for (size_t i = 0; i < means.size(); i++)
  //  delete means[i];
  //for (size_t i = 0; i < covars.size(); i++)
  //delete covars[i];
}

void FunctionApproximatorGMR::trainIncremental(const MatrixXd& inputs, const MatrixXd& targets)
{
  if (!isTrained())
  {
    //cout << " Training for the first time... " << endl;
    train(inputs,targets);
    return;
  }

  const ModelParametersGMR* model_parameters_GMR = static_cast<const ModelParametersGMR*>(getModelParameters());


  int n_gaussians = model_parameters_GMR->priors_.size();
  int n_dims_in = inputs.cols();
  int n_dims_out = targets.cols();
  int n_dims_gmm = n_dims_in + n_dims_out;

  // Initialize the means, priors and covars
  std::vector<VectorXd> means(n_gaussians);
  std::vector<MatrixXd> covars(n_gaussians);
  std::vector<double> priors(n_gaussians);
  int n_observations = 0;
  for (int i = 0; i < n_gaussians; i++)
  {
    means[i] = VectorXd(n_dims_gmm);
    priors[i] = 0.0;
    covars[i] = MatrixXd(n_dims_gmm, n_dims_gmm);
  }

  // Extract the model parameters
  for (int i = 0; i < n_gaussians; i++)
  {
    means[i].segment(0, n_dims_in)    = model_parameters_GMR->means_x_[i];
    means[i].segment(n_dims_in, n_dims_out)    = model_parameters_GMR->means_y_[i];

    covars[i].block(0, 0, n_dims_in, n_dims_in)   = model_parameters_GMR->covars_x_[i];
    covars[i].block(n_dims_in, n_dims_in, n_dims_out, n_dims_out)   = model_parameters_GMR->covars_y_[i];
    covars[i].block(n_dims_in, 0, n_dims_out, n_dims_in) = model_parameters_GMR->covars_y_x_[i];

    priors[i] = model_parameters_GMR->priors_[i];
  }
  n_observations = model_parameters_GMR->n_observations_;

  // Put the input/output data in one big matrix
  MatrixXd data = MatrixXd(inputs.rows(), n_dims_gmm);
  data << inputs, targets;

  // Expectation-Maximization Incremental
  expectationMaximizationIncremental(data, means, priors, covars, n_observations);

  // Extract the different input/output components from the means/covars which contain both
  std::vector<Eigen::VectorXd> means_x(n_gaussians);
  std::vector<Eigen::VectorXd> means_y(n_gaussians);
  std::vector<Eigen::MatrixXd> covars_x(n_gaussians);
  std::vector<Eigen::MatrixXd> covars_y(n_gaussians);
  std::vector<Eigen::MatrixXd> covars_y_x(n_gaussians);
  for (int i_gau = 0; i_gau < n_gaussians; i_gau++)
  {
    means_x[i_gau]    = means[i_gau].segment(0, n_dims_in);
    means_y[i_gau]    = means[i_gau].segment(n_dims_in, n_dims_out);

    covars_x[i_gau]   = covars[i_gau].block(0, 0, n_dims_in, n_dims_in);
    covars_y[i_gau]   = covars[i_gau].block(n_dims_in, n_dims_in, n_dims_out, n_dims_out);
    covars_y_x[i_gau] = covars[i_gau].block(n_dims_in, 0, n_dims_out, n_dims_in);
  }

  setModelParameters(new ModelParametersGMR(n_observations, priors, means_x, means_y, covars_x, covars_y, covars_y_x));

  // After training, we know the sizes of the matrices that should be cached
  preallocateMatrices(n_gaussians,n_dims_in,n_dims_out);
}

double FunctionApproximatorGMR::normalPDF(const VectorXd& mu, const MatrixXd& covar, const VectorXd& input)
{
  MatrixXd covar_inverse = covar.inverse();
  double output = exp(-0.5*(input-mu).transpose()*covar_inverse*(input-mu));
  // For invertible matrices (which covar apparently was), det(A^-1) = 1/det(A)
  // Hence the 1.0/covar_inverse.determinant() below
  //  ( (2*pi)^N*|\Sigma| )^(-1/2)
  output *= pow(pow(2*M_PI,mu.size())/covar_inverse.determinant(),-0.5);   
  return output;
}

double FunctionApproximatorGMR::normalPDFDamped(const VectorXd& mu, const MatrixXd& covar, const VectorXd& input)
{
  if(covar.determinant() > 0) // It is invertible
  {
    MatrixXd covar_inverse = covar.inverse();

      double output = exp(-0.5*(input-mu).transpose()*covar_inverse*(input-mu));

      // Check that:
      // if output == 0.0
      // return 0.0;

      // For invertible matrices (which covar apparently was), det(A^-1) = 1/det(A)
      // Hence the 1.0/covar_inverse.determinant() below
      //  ( (2\pi)^N*|\Sigma| )^(-1/2)
      output *= pow(pow(2*M_PI,mu.size())/(covar_inverse.determinant()),-0.5);
      return output;
  }
  else
  {
      //cerr << "WARNING: FunctionApproximatorGMR::normalPDFDamped output close to singularity..." << endl;
      return std::numeric_limits<double>::min();
  }
}


void FunctionApproximatorGMR::predict(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs)
{
  ENTERING_REAL_TIME_CRITICAL_CODE
  outputs.resize(inputs.rows(),getExpectedOutputDim());
  predict(inputs,outputs,empty_prealloc_);
  EXITING_REAL_TIME_CRITICAL_CODE
}

void FunctionApproximatorGMR::predictVariance(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& variances)
{
  ENTERING_REAL_TIME_CRITICAL_CODE
  variances.resize(inputs.rows(),getExpectedOutputDim());
  predict(inputs,empty_prealloc_,variances);
  EXITING_REAL_TIME_CRITICAL_CODE
}

void FunctionApproximatorGMR::predict(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs, Eigen::MatrixXd& variances)
{
  ENTERING_REAL_TIME_CRITICAL_CODE
  
  // The reason this function is not pretty and so long (which I usually try to avoid) is to  
  // avoid Eigen making dynamic memory allocations. This would cause trouble in real-time critical
  // code. Therefore, I
  //   * use preallocated matrices (member variables) for intermediate results
  //   * use noalias() whenever needed (http://eigen.tuxfamily.org/dox/TopicLazyEvaluation.html)
  //   * try to avoid calling other functions (a bit tricky when using Eigen matrices as parameters)
  // I hope the documentation makes up for the ugliness...
  
  if (!isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorGMR::predict if you have not trained yet. Doing nothing." << endl;
    return;
  }
  
  const ModelParametersGMR* gmm = static_cast<const ModelParametersGMR*>(getModelParameters());

  // Number of Gaussians must be at least one
  assert(gmm->getNumberOfGaussians()>0);
  // Dimensionality of input must be same as of the gmm inputs  
  assert(gmm->getExpectedInputDim()==inputs.cols());

  // Only compute the means if the outputs matrix is not empty
  bool compute_means = false;
  if (outputs.rows()>0)
  {
    outputs.resize(inputs.rows(),gmm->getExpectedOutputDim());
    outputs.fill(0);
    compute_means = true;
  }
  
  // Only compute the variances if the variances matrix is not empty
  bool compute_variance = false;
  if (variances.rows()>0)
  {
    compute_variance = true;
    variances.resize(inputs.rows(),gmm->getExpectedOutputDim());
    variances.fill(0);
  }

  // For each input, compute the output  
  for (int i_input=0; i_input<inputs.rows(); i_input++)
  {
    
    // Three main steps
    // A: compute probability: prior * pdf of the multivariate Gaussian
    // B: compute estimated mean of y: (mu_y + ( C_y_x * inv(C_x) * (input-mu_x) ) )
    // C: weight the estimated mean with the probability
    
    
    // A: compute probability: prior * pdf of the multivariate Gaussian
    // Compute probalities that each Gaussian would generate this input in 4 steps
    // A1. Compute the unnormalized pdf of the multi-variate Gaussian distribution
    // A2. Normalize the unnormalized pdf (scale factor has been precomputed in the GMM)
    // A3. Multiply the normalized pdf with the priors
    // A4. Normalize the probabilities by dividing by their sum
  
    for (unsigned int i_gau=0; i_gau<gmm->getNumberOfGaussians(); i_gau++)
    {
      // A1. Compute the unnormalized pdf of the multi-variate Gaussian distribution
      // formula: exp( -2 * (x-mu)^T * Sigma^-1 * (x-mu) )
      // (we use cached variables and noalias to avoid dynamic allocation)
      // (x-mu)
      diff_prealloc_ = inputs.row(i_input).transpose() - gmm->means_x_[i_gau];
      // Sigma^-1 * (x-mu)
      covar_times_diff_prealloc_.noalias() = gmm->covars_x_inv_[i_gau]*diff_prealloc_;
      // exp( -2 * (x-mu)^T * Sigma^-1 * (x-mu) )
      probabilities_prealloc_[i_gau] = exp(-0.5*diff_prealloc_.dot(covar_times_diff_prealloc_));
      
      // A2. Normalize the unnormalized pdf (scale factor has been precomputed in the GMM)
      // formula for scale factor: 1/sqrt( (2\pi)^N*|\Sigma| )
      probabilities_prealloc_[i_gau] *= gmm->mvgd_scale_[i_gau];
      
      // A3. Multiply the normalized pdf with the priors
      probabilities_prealloc_[i_gau] *= gmm->priors_[i_gau];
      
    }
    
    // A4. Normalize the probabilities by dividing by their sum
    probabilities_prealloc_ /= probabilities_prealloc_.sum();
    

    if (compute_means)
    {
      for (unsigned int i_gau=0; i_gau<gmm->getNumberOfGaussians(); i_gau++)
      {
        
        // B: compute estimated mean of y: (mu_y + ( C_y_x * inv(C_x) * (input-mu_x) ) )
        // We will compute it bit by bit (with preallocated matrices) to avoid dynamic allocations.
        
        // (input-mu_x)
        diff_prealloc_ = inputs.row(i_input).transpose() - gmm->means_x_[i_gau];
        // inv(C_x) * (input-mu_x)
        covar_times_diff_prealloc_.noalias() = gmm->covars_x_inv_[i_gau]*diff_prealloc_;
        // ( C_y_x * inv(C_x) * (input-mu_x) )
        mean_output_prealloc_.noalias() = gmm->covars_y_x_[i_gau]*covar_times_diff_prealloc_;
        // (mu_y + ( C_y_x * inv(C_x) * (input-mu_x) ) )
        mean_output_prealloc_ += gmm->means_y_[i_gau];
        
        // C: weight the estimated mean with the probability
        // probability * (mu_y + ( C_y_x * inv(C_x) * (input-mu_x) ) )
        outputs.row(i_input) += probabilities_prealloc_[i_gau] * mean_output_prealloc_; 
      }
    }
   
    if (compute_variance)
    {
      for (unsigned int i_gau=0; i_gau<gmm->getNumberOfGaussians(); i_gau++)
      {
        // Here comes the formula: h^2 * (C_y - C_y_x * inv(C_x) * C_y_x^T) 
        
        // inv(C_x) * C_y_x^T
        covar_input_times_output_.noalias() = gmm->covars_x_inv_[i_gau]*gmm->covars_y_x_[i_gau].transpose();
        // - C_y_x * inv(C_x) * C_y_x^T
        covar_output_prealloc_.noalias() = gmm->covars_y_x_[i_gau] * covar_input_times_output_;
        // NOTE: covar_output_prealloc_.noalias() = - gmm->covars_y_x_[i_gau] * covar_input_times_output_; causes memory allocation
        // so we split it in two operations
        covar_output_prealloc_.noalias() = - covar_output_prealloc_;
        // (C_y - C_y_x * inv(C_x) * C_y_x^T) 
        covar_output_prealloc_ += gmm->covars_y_[i_gau];
        // h^2 * (C_y - C_y_x * inv(C_x) * C_y_x^T) 
        variances.row(i_input) += probabilities_prealloc_[i_gau]*probabilities_prealloc_[i_gau] * ( covar_output_prealloc_ ).diagonal();
        
        // There are cases where we may get slightly negative variances due to numerical issues
        // Avoid them here by setting negative variances to 0.
        for (int i_output_dim=0; i_output_dim<gmm->getExpectedOutputDim(); i_output_dim++)
          if (variances(i_input,i_output_dim)<0.0)
            variances(i_input,i_output_dim) = 0.0;


      }
    }
  }
  
  EXITING_REAL_TIME_CRITICAL_CODE
}

void FunctionApproximatorGMR::predictDot(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs, Eigen::MatrixXd& outputs_dot)
{
  ENTERING_REAL_TIME_CRITICAL_CODE
  outputs.resize(inputs.rows(),getExpectedOutputDim());
  predictDot(inputs,outputs,outputs_dot,empty_prealloc_);
  EXITING_REAL_TIME_CRITICAL_CODE
}

void FunctionApproximatorGMR::predictDot(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs, Eigen::MatrixXd& outputs_dot, Eigen::MatrixXd& variances)
{
  ENTERING_REAL_TIME_CRITICAL_CODE
 
  // The reason this function is not pretty and so long (which I usually try to avoid) is to  
  // avoid Eigen making dynamic memory allocations. This would cause trouble in real-time critical
  // code. Therefore, I
  //   * use preallocated matrices (member variables) for intermediate results
  //   * use noalias() whenever needed (http://eigen.tuxfamily.org/dox/TopicLazyEvaluation.html)
  //   * try to avoid calling other functions (a bit tricky when using Eigen matrices as parameters)
  // I hope the documentation makes up for the ugliness...
  
  if (!isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorGMR::predict if you have not trained yet. Doing nothing." << endl;
    return;
  }
  
  const ModelParametersGMR* gmm = static_cast<const ModelParametersGMR*>(getModelParameters());

  // Dimensionality of input must be 1 in order to compute the derivative
  assert(inputs.cols() == 1); 
  // Number of Gaussians must be at least one
  assert(gmm->getNumberOfGaussians()>0);
  // Dimensionality of input must be same as of the gmm inputs  
  assert(gmm->getExpectedInputDim()==inputs.cols());

  // Only compute the means if the outputs matrix is not empty
  bool compute_means = false;
  if (outputs.rows()>0)
  {
    outputs.resize(inputs.rows(),gmm->getExpectedOutputDim());
    outputs_dot.resize(inputs.rows(),gmm->getExpectedOutputDim());
    outputs.fill(0);
    outputs_dot.fill(0);
    compute_means = true;
  }
  
  // Only compute the variances if the variances matrix is not empty
  bool compute_variance = false;
  if (variances.rows()>0)
  {
    compute_variance = true;
    variances.resize(inputs.rows(),gmm->getExpectedOutputDim());
    variances.fill(0);
  }

  // For each input, compute the output  
  for (int i_input=0; i_input<inputs.rows(); i_input++)
  {
    
    // Three main steps
    // A: compute probability: prior * pdf of the multivariate Gaussian
    // B: compute estimated mean of y: (mu_y + ( C_y_x * inv(C_x) * (input-mu_x) ) )
    // C: weight the estimated mean with the probability
    
    
    // A: compute probability: prior * pdf of the multivariate Gaussian
    // Compute probalities that each Gaussian would generate this input in 4 steps
    // A1. Compute the unnormalized pdf of the multi-variate Gaussian distribution
    // A2. Normalize the unnormalized pdf (scale factor has been precomputed in the GMM)
    // A3. Multiply the normalized pdf with the priors
    // A4. Normalize the probabilities by dividing by their sum
  
    for (unsigned int i_gau=0; i_gau<gmm->getNumberOfGaussians(); i_gau++)
    {
      // A1. Compute the unnormalized pdf of the multi-variate Gaussian distribution
      // formula: exp( -2 * (x-mu)^T * Sigma^-1 * (x-mu) )
      // (we use cached variables and noalias to avoid dynamic allocation)
      // (x-mu)
      diff_prealloc_ = inputs.row(i_input).transpose() - gmm->means_x_[i_gau];
      // Sigma^-1 * (x-mu)
      covar_times_diff_prealloc_.noalias() = gmm->covars_x_inv_[i_gau]*diff_prealloc_;
      // exp( -2 * (x-mu)^T * Sigma^-1 * (x-mu) )
      probabilities_prealloc_[i_gau] = exp(-0.5*diff_prealloc_.dot(covar_times_diff_prealloc_));
      
      // A2. Normalize the unnormalized pdf (scale factor has been precomputed in the GMM)
      // formula for scale factor: 1/sqrt( (2\pi)^N*|\Sigma| )
      probabilities_prealloc_[i_gau] *= gmm->mvgd_scale_[i_gau];
      
      // A3. This is the derivate of an exponential function, NOTE that we are assuming input dimension equal to one!
      probabilities_dot_prealloc_[i_gau] *= - probabilities_prealloc_[i_gau] * covar_times_diff_prealloc_(0,0);
      
      // A4. Multiply the normalized pdf with the priors
      probabilities_prealloc_[i_gau] *= gmm->priors_[i_gau];
      
      // A5. Repeat the same with the prob. dot
      probabilities_dot_prealloc_[i_gau] *= gmm->priors_[i_gau];
      
    }
    
    // A5. Normalize the probabilities by dividing by their sum
    probabilities_prealloc_sum_ = probabilities_prealloc_.sum();
    probabilities_prealloc_ /= probabilities_prealloc_sum_;
    
    // A6. Compute the derivative of the probability
    probabilities_dot_prealloc_sum_ = probabilities_dot_prealloc_.sum();
    probabilities_dot_prealloc_ = (probabilities_dot_prealloc_ * probabilities_prealloc_sum_ - probabilities_prealloc_ * probabilities_dot_prealloc_sum_)/pow(probabilities_prealloc_sum_,2);

    if (compute_means)
    {
      for (unsigned int i_gau=0; i_gau<gmm->getNumberOfGaussians(); i_gau++)
      {
        
        // B: compute estimated mean of y: (mu_y + ( C_y_x * inv(C_x) * (input-mu_x) ) )
        // We will compute it bit by bit (with preallocated matrices) to avoid dynamic allocations.
        // (input-mu_x)
        diff_prealloc_ = inputs.row(i_input).transpose() - gmm->means_x_[i_gau];
        // inv(C_x) * (input-mu_x)
        covar_times_diff_prealloc_.noalias() = gmm->covars_x_inv_[i_gau]*diff_prealloc_;
        // ( C_y_x * inv(C_x) * (input-mu_x) )
        mean_output_prealloc_.noalias() = gmm->covars_y_x_[i_gau]*covar_times_diff_prealloc_;
        // (mu_y + ( C_y_x * inv(C_x) * (input-mu_x) ) )
        mean_output_prealloc_ += gmm->means_y_[i_gau];
        
        // C: weight the estimated mean with the probability
        // probability * (mu_y + ( C_y_x * inv(C_x) * (input-mu_x) ) )
        outputs.row(i_input) += probabilities_prealloc_[i_gau] * mean_output_prealloc_; 
	
	// D: compute the derivate of the output
        // (C_y_x * inv(C_x))
	covar_output_times_input_.noalias() = gmm->covars_y_x_[i_gau] * gmm->covars_x_inv_[i_gau];
	// probability_dot * (mu_y + ( C_y_x * inv(C_x) * (input-mu_x) ) ) + probability * (C_y_x * inv(C_x))
	outputs_dot.row(i_input) += probabilities_dot_prealloc_[i_gau] * mean_output_prealloc_ +  probabilities_prealloc_[i_gau] * covar_output_times_input_;
	
      }
    }
   
    if (compute_variance)
    {
      for (unsigned int i_gau=0; i_gau<gmm->getNumberOfGaussians(); i_gau++)
      {
        // Here comes the formula: h^2 * (C_y - C_y_x * inv(C_x) * C_y_x^T) 
        
        // inv(C_x) * C_y_x^T
        covar_input_times_output_.noalias() = gmm->covars_x_inv_[i_gau]*gmm->covars_y_x_[i_gau].transpose();
        // - C_y_x * inv(C_x) * C_y_x^T
        covar_output_prealloc_.noalias() = gmm->covars_y_x_[i_gau] * covar_input_times_output_;
        // NOTE: covar_output_prealloc_.noalias() = - gmm->covars_y_x_[i_gau] * covar_input_times_output_; causes memory allocation
        // so we split it in two operations
        covar_output_prealloc_.noalias() = - covar_output_prealloc_;
        // (C_y - C_y_x * inv(C_x) * C_y_x^T) 
        covar_output_prealloc_ += gmm->covars_y_[i_gau];
        // h^2 * (C_y - C_y_x * inv(C_x) * C_y_x^T) 
        variances.row(i_input) += probabilities_prealloc_[i_gau]*probabilities_prealloc_[i_gau] * ( covar_output_prealloc_ ).diagonal();
        
        // There are cases where we may get slightly negative variances due to numerical issues
        // Avoid them here by setting negative variances to 0.
        for (int i_output_dim=0; i_output_dim<gmm->getExpectedOutputDim(); i_output_dim++)
          if (variances(i_input,i_output_dim)<0.0)
            variances(i_input,i_output_dim) = 0.0;


      }
    }
  }
  
  EXITING_REAL_TIME_CRITICAL_CODE
}

void FunctionApproximatorGMR::firstDimSlicingInit(const MatrixXd& data, std::vector<VectorXd>& centers, std::vector<double>& priors,
  std::vector<MatrixXd>& covars)
{

  VectorXd first_dim = data.col(0);

  VectorXi assign(data.rows());
  assign.setZero();

  double min_val = first_dim.minCoeff();
  double max_val = first_dim.maxCoeff();

  for (int i_first_dim = 0; i_first_dim < first_dim.size(); i_first_dim++)
  {
    unsigned int center = int((first_dim[i_first_dim]-min_val)/(max_val-min_val)*centers.size());
    if (center==centers.size())
      center--;
    assign[i_first_dim] = center;
  }
  
  // Init means
  VectorXi nbPoints = VectorXi::Zero(centers.size());
  for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
    centers[i_gau].setZero();
  for (int iData = 0; iData < data.rows(); iData++)
  {
    centers[assign[iData]] += data.row(iData).transpose();
    nbPoints[assign[iData]]++;
  }
  for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
    centers[i_gau] /= nbPoints[i_gau];

  // Init covars
  for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
    covars[i_gau].setZero();
  for (int iData = 0; iData < data.rows(); iData++)
    covars[assign[iData]] += (data.row(iData).transpose() - centers[assign[iData]]) * (data.row(iData).transpose() - centers[assign[iData]]).transpose();
  for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
    covars[i_gau] /= nbPoints[i_gau];

  // Be sure that covar is invertible
  for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
      covars[i_gau] += MatrixXd::Identity(covars[i_gau].rows(), covars[i_gau].cols()) * 1e-5;

  // Init priors
  for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
    priors[i_gau] = 1. / centers.size();
}

void FunctionApproximatorGMR::kMeansInit(const MatrixXd& data, std::vector<VectorXd>& centers, std::vector<double>& priors,
  std::vector<MatrixXd>& covars, int n_max_iter)
{

  MatrixXd dataCentered = data.rowwise() - data.colwise().mean();
  MatrixXd dataCov = dataCentered.transpose() * dataCentered / data.rows();
  MatrixXd dataCovInverse = dataCov.inverse();

  std::vector<int> dataIndex;
  for (int i = 0; i < data.rows(); i++)
    dataIndex.push_back(i); 
  std::random_shuffle (dataIndex.begin(), dataIndex.end());

  for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
    centers[i_gau] = data.row(dataIndex[i_gau]);

  VectorXi assign(data.rows());
  assign.setZero();

  bool converged = false;
  for (int iIter = 0; iIter < n_max_iter && !converged; iIter++)
  {
    //cout << "  iIter=" << iIter << endl;
    
    // E step
    converged = true;
    for (int iData = 0; iData < data.rows(); iData++)
    {
      VectorXd v = (centers[assign[iData]] - data.row(iData).transpose());

      double minDist = v.transpose() * dataCovInverse * v;

      for (int i_gau = 0; i_gau < (int)centers.size(); i_gau++)
      {
        if (i_gau == assign[iData])
          continue;

        v = (centers[i_gau] - data.row(iData).transpose());
        double dist = v.transpose() * dataCovInverse * v;
        if (dist < minDist)
        {
          converged = false;
          minDist = dist;
          assign[iData] = i_gau;
        }
      }
    }

    // M step
    VectorXi nbPoints = VectorXi::Zero(centers.size());
    for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
      centers[i_gau].setZero();
    for (int iData = 0; iData < data.rows(); iData++)
    {
      centers[assign[iData]] += data.row(iData).transpose();
      nbPoints[assign[iData]]++;
    }
    for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
      centers[i_gau] /= nbPoints[i_gau];
  }

  // Init covars
  VectorXi nbPoints = VectorXi::Zero(centers.size());
  for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
    covars[i_gau].setZero();
  for (int iData = 0; iData < data.rows(); iData++)
  {
    covars[assign[iData]] += (data.row(iData).transpose() - centers[assign[iData]]) * (data.row(iData).transpose() - centers[assign[iData]]).transpose();
    nbPoints[assign[iData]]++;
  }
  for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
    covars[i_gau] /= nbPoints[i_gau];

  // Be sure that covar is invertible
  for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
    covars[i_gau] += MatrixXd::Identity(covars[i_gau].rows(), covars[i_gau].cols()) * 1e-5f;

  // Init priors
  for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
    priors[i_gau] = 1. / centers.size();
}

void FunctionApproximatorGMR::expectationMaximization(const MatrixXd& data, std::vector<VectorXd>& centers, std::vector<double>& priors,
    std::vector<MatrixXd>& covars, int n_max_iter)
{
  MatrixXd assign(centers.size(), data.rows());
  assign.setZero();

  std::vector<double> E(centers.size());

  double oldLoglik = -1e10f;
  double loglik = 0;

  for (int iIter = 0; iIter < n_max_iter; iIter++)
  {
    //cout << "  iIter=" << iIter << endl;
    // For debugging only
    //ModelParametersGMR::saveGMM("/tmp/demoTrainFunctionApproximators/GMR",centers,covars,iIter);
    
    // E step
    for (int iData = 0; iData < data.rows(); iData++)
      for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
        assign(i_gau, iData) = priors[i_gau] * FunctionApproximatorGMR::normalPDF(centers[i_gau], covars[i_gau],data.row(iData).transpose());

    oldLoglik = loglik;
    loglik = 0;
    double sum_tmp = 0.0;
    for (int iData = 0; iData < data.rows(); iData++)
    {
        sum_tmp = assign.col(iData).sum();
        loglik += log(sum_tmp);
    }
    loglik /= data.rows();

    if (fabs(loglik / oldLoglik - 1) < 1e-8f)
      break;

    for (int iData = 0; iData < data.rows(); iData++)
      assign.col(iData) /= assign.col(iData).sum();

    if (fabs(loglik / oldLoglik - 1) < 1e-8f)
      break;

    // M step
    for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
    {
      centers[i_gau].setZero();
      covars[i_gau].setZero();
      priors[i_gau] = 0;
      E[i_gau] = assign.row(i_gau).sum();
    }

    for (int iData = 0; iData < data.rows(); iData++)
    {
      for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
      {
        centers[i_gau] += assign(i_gau, iData) * data.row(iData).transpose();
        priors[i_gau] += assign(i_gau, iData);
      }
    }

    for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
    {
      centers[i_gau] /= assign.row(i_gau).sum();
      priors[i_gau] /= assign.cols();
    }

    for (int iData = 0; iData < data.rows(); iData++)
      for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
        covars[i_gau] += assign(i_gau, iData) * (data.row(iData).transpose() - centers[i_gau]) * (data.row(iData).transpose() - centers[i_gau]).transpose();

    for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
      covars[i_gau] /= assign.row(i_gau).sum();

    // Be sure that covar is invertible
    for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
      covars[i_gau] += MatrixXd::Identity(covars[i_gau].rows(), covars[i_gau].cols()) * 1e-5f;
  }
  
  /*
  Here's a hacky Matlab script for plotting the EM procedure above (if you cann saveGMM)
  directory = '/tmp/demoTrainFunctionApproximators/GMR/';
  inputs = load([directory '/inputs.txt']);
  targets = load([directory '/targets.txt']);
  outputs = load([directory '/outputs.txt']);
   
  
  plot(inputs,targets,'.k')
  hold on
  plot(inputs,outputs,'.r')
  
  max_iter = 5;
  for iter=0:max_iter
    color = 0.2+(0.8-iter/max_iter)*[1 1 1];
    for bfs=0:2
      center = load(sprintf('%s/gmm_iter%02d_mu%03d.txt',directory,iter,bfs));
      %plot(center(1),center(2))
      covar = load(sprintf('%s/gmm_iter%02d_covar%03d.txt',directory,iter,bfs));
      h = error_ellipse(covar,center,'conf',0.95);
      set(h,'Color',color,'LineWidth',1+iter/max_iter)
    end
  end
  hold off
  */
  
}

void FunctionApproximatorGMR::expectationMaximizationIncremental(const MatrixXd& data, std::vector<VectorXd>& centers, std::vector<double>& priors,
    std::vector<MatrixXd>& covars, int& n_observations, int n_max_iter)
{

  std::vector<VectorXd> centers_prev = centers;
  std::vector<double> priors_prev = priors;
  std::vector<MatrixXd> covars_prev = covars;
  int n_observations_prev = n_observations;
  std::vector<double> E_prev;
  std::vector<double> E(centers.size());
  n_observations = data.rows();

  double oldLoglik = -1e10f;
  double loglik = 0;

  MatrixXd assign(centers.size(), data.rows());
  assign.setZero();

  // Compute the old E
  E_prev.resize(centers.size());
  for (size_t i_gau = 0; i_gau<centers.size(); i_gau++)
      E_prev[i_gau] = priors_prev[i_gau] * n_observations_prev;

  for (int iIter = 0; iIter < n_max_iter; iIter++)
  {
    //cout << "  iIter=" << iIter << endl;
    // For debugging only
    //ModelParametersGMR::saveGMM("/tmp/demoTrainFunctionApproximators/GMR",centers,covars,iIter);

    // E step
    for (int iData = 0; iData < data.rows(); iData++)
      for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
        assign(i_gau, iData) = priors[i_gau] * FunctionApproximatorGMR::normalPDFDamped(centers[i_gau], covars[i_gau],data.row(iData).transpose());

    oldLoglik = loglik;
    loglik = 0;
    double sum_tmp = 0.0;
    for (int iData = 0; iData < data.rows(); iData++)
    {
        sum_tmp = assign.col(iData).sum();
        loglik += log(sum_tmp);
    }
    loglik /= data.rows();

    for (int iData = 0; iData < data.rows(); iData++)
      assign.col(iData) /= assign.col(iData).sum();

    if (fabs(loglik / oldLoglik - 1) < 1e-8f)
      break;

    // M step
    for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
    {
      centers[i_gau].setZero();
      covars[i_gau].setZero();
      priors[i_gau] = 0;
      E[i_gau] = assign.row(i_gau).sum();
      //E_prev[i_gau] = assign_prev.row(i_gau).sum();
    }

    for (int iData = 0; iData < data.rows(); iData++)
    {
      for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
      {
          centers[i_gau] += assign(i_gau, iData) * data.row(iData).transpose();
      }
    }

    for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
    {
      centers[i_gau] = (E_prev[i_gau] * centers_prev[i_gau] + centers[i_gau])/(E[i_gau] + E_prev[i_gau]);
      priors[i_gau] = (E[i_gau] + E_prev[i_gau])/(n_observations + n_observations_prev);
    }

    for (int iData = 0; iData < data.rows(); iData++)
      for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
        covars[i_gau] += assign(i_gau, iData) * (data.row(iData).transpose() - centers[i_gau]) * (data.row(iData).transpose() - centers[i_gau]).transpose();

    for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
      covars[i_gau] = covars[i_gau] +  E_prev[i_gau] * (covars_prev[i_gau] + (centers_prev[i_gau] - centers[i_gau]) * (centers_prev[i_gau] - centers[i_gau]).transpose());

    for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
        covars[i_gau] /= (E[i_gau] + E_prev[i_gau]);

    // Be sure that covar is invertible
    for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
      covars[i_gau] += MatrixXd::Identity(covars[i_gau].rows(), covars[i_gau].cols()) * 1e-5f;
  }

  // Increase the total number of obs counting the old plus the new
  n_observations += n_observations_prev;
}

template<class Archive>
void FunctionApproximatorGMR::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(FunctionApproximator);
}



}
