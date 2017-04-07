/**
 * @file FunctionApproximatorGMR.hpp
 * @brief FunctionApproximatorGMR class header file.
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

#ifndef _FUNCTION_APPROXIMATOR_GMR_H_
#define _FUNCTION_APPROXIMATOR_GMR_H_

#include "functionapproximators/FunctionApproximator.hpp"


/** @defgroup GMR Gaussian Mixture Regression (GMR)
 *  @ingroup FunctionApproximators
 */
 
namespace DmpBbo {

  // Forward declarations
class MetaParametersGMR;
class ModelParametersGMR;

/** \brief GMR (Gaussian Mixture Regression) function approximator
 * \ingroup FunctionApproximators
 * \ingroup GMR
 */
class FunctionApproximatorGMR : public FunctionApproximator
{
public:
  /** Initialize a function approximator with meta- and model-parameters
   *  \param[in] meta_parameters  The training algorithm meta-parameters
   *  \param[in] model_parameters The parameters of the trained model. If this parameter is not
   *                              passed, the function approximator is initialized as untrained. 
   *                              In this case, you must call FunctionApproximator::train() before
   *                              being able to call FunctionApproximator::predict().
   * Either meta_parameters XOR model-parameters can passed as NULL, but not both.
   */
  FunctionApproximatorGMR(const MetaParametersGMR *const meta_parameters, const ModelParametersGMR *const model_parameters=NULL);  

  /** Initialize a function approximator with model parameters
   *  \param[in] model_parameters The parameters of the (previously) trained model.
   */
  FunctionApproximatorGMR(const ModelParametersGMR *const model_parameters);

	virtual FunctionApproximator* clone(void) const;
	
	void train(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target);
  
	void predict(const Eigen::MatrixXd& input, Eigen::MatrixXd& output);

	void predictVariance(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& variances);

	void predict(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs, Eigen::MatrixXd& variances);
	
	std::string getName(void) const {
    return std::string("GMR");  
  };

protected:
  /** Initialize Gaussian for EM algorithm using k-means. 
   * \param[in]  data A data matrix (n_exemples x (n_in_dim + n_out_dim))
   * \param[out]  means A list (std::vector) of n_gaussian non initiallized means (n_in_dim + n_out_dim)
   * \param[out]  priors A list (std::vector) of n_gaussian non initiallized priors
   * \param[out]  covars A list (std::vector) of n_gaussian non initiallized covariance matrices ((n_in_dim + n_out_dim) x (n_in_dim + n_out_dim))
   * \param[in]  n_max_iter The maximum number of iterations
   * \author Thibaut Munzer
   */
  void kMeansInit(const Eigen::MatrixXd& data, std::vector<Eigen::VectorXd>& means, std::vector<double>& priors,
    std::vector<Eigen::MatrixXd>& covars, int n_max_iter=1000);

  /** Initialize Gaussian for EM algorithm using a same-size slicing on the first dimension (method used in Calinon GMR implementation).
   * Particulary suited when input is 1-D and data distribution is uniform over input dimension
   * \param[in]  data A data matrix (n_exemples x (n_in_dim + n_out_dim))
   * \param[out]  means A list (std::vector) of n_gaussian non initiallized means (n_in_dim + n_out_dim)
   * \param[out]  priors A list (std::vector) of n_gaussian non initiallized priors
   * \param[out]  covars A list (std::vector) of n_gaussian non initiallized covariance matrices ((n_in_dim + n_out_dim) x (n_in_dim + n_out_dim))
   * \author Thibaut Munzer
   */
  void firstDimSlicingInit(const Eigen::MatrixXd& data, std::vector<Eigen::VectorXd>& means, std::vector<double>& priors,
    std::vector<Eigen::MatrixXd>& covars);

  /** EM algorithm. 
   * \param[in] data A (n_exemples x (n_in_dim + n_out_dim)) data matrix
   * \param[in,out] means A list (std::vector) of n_gaussian means (vector of size (n_in_dim + n_out_dim))
   * \param[in,out] priors A list (std::vector) of n_gaussian priors
   * \param[in,out] covars A list (std::vector) of n_gaussian covariance matrices ((n_in_dim + n_out_dim) x (n_in_dim + n_out_dim))
   * \param[in] n_max_iter The maximum number of iterations
   * \author Thibaut Munzer
   */
  void expectationMaximization(const Eigen::MatrixXd& data, std::vector<Eigen::VectorXd>& means, std::vector<double>& priors,
    std::vector<Eigen::MatrixXd>& covars, int n_max_iter=50);
  
  /** EM algorithm Incremental.
   * \param[in] data A (n_exemples x (n_in_dim + n_out_dim)) data matrix
   * \param[in,out] means A list (std::vector) of n_gaussian means (vector of size (n_in_dim + n_out_dim))
   * \param[in,out] priors A list (std::vector) of n_gaussian priors
   * \param[in,out] covars A list (std::vector) of n_gaussian covariance matrices ((n_in_dim + n_out_dim) x (n_in_dim + n_out_dim))
   * \param[in,out] n_observations Number of observations
   * \param[in] n_max_iter The maximum number of iterations
   * \author Gennaro Raiola
   */
  void expectationMaximizationIncremental(const Eigen::MatrixXd& data, std::vector<Eigen::VectorXd>& means, std::vector<double>& priors,
    std::vector<Eigen::MatrixXd>& covars, int& n_observations, int n_max_iter=50);

  /** The probability density function (PDF) of the multi-variate normal distribution
   * \param[in] mu The mean of the normal distribution
   * \param[in] covar The covariance matrix of the normal distribution 
   * \param[in] input The input data vector for which the PDF will be computed.
   * \return The PDF value for the input
   */
  static double normalPDF(const Eigen::VectorXd& mu, const Eigen::MatrixXd& covar, const Eigen::VectorXd& input);

  static double normalPDFDamped(const Eigen::VectorXd& mu, const Eigen::MatrixXd& covar, const Eigen::VectorXd& input);

public:
   /** Query the function approximator to make a prediction and to compute the derivate of that prediction
   *  \param[in]  inputs   Input values of the query
   *  \param[out] outputs  Predicted output values
   *  \param[out] outputs_dot  Predicted derivate values
   * 
   * \remark This method should be const. But third party functions which is called in this function
   * have not always been implemented as const (Examples: LWPRObject::predict or IRFRLS::predict ).
   * Therefore, this function cannot be const.
   */
  void predictDot(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs, Eigen::MatrixXd& outputs_dot);
  
  /** Query the function approximator to make a prediction and to compute the derivate of that prediction, and also to predict its variance
   *  \param[in]  inputs   Input values of the query (n_samples X n_dims_in)
   *  \param[out] outputs  Predicted output values (n_samples X n_dims_out)
   *  \param[out] outputs_dot  Predicted derivate values
   *  \param[out] variances Predicted variances for the output values  (n_samples X n_dims_out). Note that if the output has a dimensionality>1, these variances should actuall be covariance matrices (use function predict(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs, std::vector<Eigen::MatrixXd>& variances) to get the full covariance matrices). So for an output dimensionality of 1 this function works fine. For dimensionality>1 we return only the diagional of the covariance matrix, which may not always be what you want.
   *
   * \remark This method should be const. But third party functions which is called in this function
   * have not always been implemented as const (Examples: LWPRObject::predict or IRFRLS::predict ).
   * Therefore, this function cannot be const.
   */
  void predictDot(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs, Eigen::MatrixXd& outputs_dot, Eigen::MatrixXd& variances);

  void trainIncremental(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets);


private:
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  FunctionApproximatorGMR(void) {};
  
  /** Give boost serialization access to private members. */  
  friend class boost::serialization::access;
  
  /** Serialize class data members to boost archive. 
   * \param[in] ar Boost archive
   * \param[in] version Version of the class
   * See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/tutorial.html#simplecase
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version);
  
  void preallocateMatrices(int n_gaussians, int n_input_dims, int n_output_dims);
  /** This is a cached variable whose memory is allocated once during construction. */
  Eigen::VectorXd probabilities_prealloc_;
  Eigen::VectorXd diff_prealloc_;
  Eigen::VectorXd covar_times_diff_prealloc_;
  Eigen::VectorXd mean_output_prealloc_;
  Eigen::MatrixXd covar_input_times_output_;
  Eigen::MatrixXd covar_output_times_input_;
  Eigen::MatrixXd covar_output_prealloc_;
  Eigen::VectorXd probabilities_dot_prealloc_;
  Eigen::MatrixXd empty_prealloc_;
  double probabilities_prealloc_sum_;
  double probabilities_dot_prealloc_sum_;
};

}

#include <boost/serialization/export.hpp>
/** Register this derived class. */
BOOST_CLASS_EXPORT_KEY2(DmpBbo::FunctionApproximatorGMR, "FunctionApproximatorGMR")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::FunctionApproximatorGMR,boost::serialization::object_serializable);

#endif // !_FUNCTION_APPROXIMATOR_GMR_H_
