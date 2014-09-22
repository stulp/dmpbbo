/**
 * @file   ModelParametersGPR.hpp
 * @brief  ModelParametersGPR class header file.
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
 
#ifndef MODELPARAMETERSGPR_H
#define MODELPARAMETERSGPR_H

#include "functionapproximators/ModelParameters.hpp"

#include <iosfwd>
#include <vector>

#include <eigen3/Eigen/Core>

namespace DmpBbo {

/** \brief Model parameters for the Gaussian Process Regression (GPR) function approximator
 * \ingroup FunctionApproximators
 * \ingroup GPR
 */
class ModelParametersGPR : public ModelParameters
{
  friend class FunctionApproximatorGPR;
  
public:
  /** Constructor for the model parameters of the GPR function approximator.
   *  \param[in] train_inputs The training samples provided (input values) 
   *  \param[in] train_targets The training samples provided (target values) 
   *  \param[in] gram The Gram matrix, i.e. the covariances between all combination of input samples 
   *  \param[in] maximum_covariance The maximum allowable covariance of the covar function (aka sigma)
   *  \param[in] length             Length 'l' of the isotropic Gaussian covariance function, i.e. sigma^2 exp(-(x-x')^2/2l^2)
   */
   ModelParametersGPR(const Eigen::MatrixXd& train_inputs, const Eigen::VectorXd& train_targets, const Eigen::MatrixXd& gram, double maximum_covariance, double length);

  /** Constructor for the model parameters of the GPR function approximator.
   *  \param[in] train_inputs The training samples provided (input values) 
   *  \param[in] train_targets The training samples provided (target values) 
   *  \param[in] gram The Gram matrix, i.e. the covariances between all combination of input samples 
   *  \param[in] maximum_covariance The maximum allowable covariance of the covar function (aka sigma)
   *  \param[in] sigmas Standard deviations, i.e. on the diagonal of the covariance matrices of the Gaussian covariance function.
   */
   ModelParametersGPR(const Eigen::MatrixXd& train_inputs, const Eigen::VectorXd& train_targets, const Eigen::MatrixXd& gram, double maximum_covariance, const Eigen::VectorXd& sigmas);
   
  std::string toString(void) const;
  
	ModelParameters* clone(void) const;
	
  int getExpectedInputDim(void) const  {
    return train_inputs_.cols();
  };
  
  
  void getSelectableParameters(std::set<std::string>& selected_values_labels) const;
  void getParameterVectorMask(const std::set<std::string> selected_values_labels, Eigen::VectorXi& selected_mask) const;
  void getParameterVectorAll(Eigen::VectorXd& all_values) const;
  inline int getParameterVectorAllSize(void) const
  {
    return 0;
  }
  
  /** Get the normalized kernel activations for given inputs
   * \param[in] inputs The input data (size: n_samples X n_dims)
   * \param[out] kernel_activations The normalized kernel activations, computed for each of the sampels in the input data (size: n_samples X n_basis_functions)
   */
  void kernelActivations(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& kernel_activations) const;

  UnifiedModel* toUnifiedModel(void) const;
  
  /** Get the maximum covariance value of the covariance function.
   * \return Maximum covariance  
   */
  double maximum_covariance(void) const { return maximum_covariance_; } ;
  
  /**
   * Return the weights, which is equivalent to \f$ \mathbf{G}^-1\mathfb{y} \f$, where G is the Gram matrix, and y is the target data.
   * \return Weights
   */
  const Eigen::VectorXd& weights(void) const { return gram_inv_targets_; };
  
  /**
   * Return the inverse of the Gram matrix
   * \return Inverse of the Gram matrix
   */
  const Eigen::MatrixXd& gram_inv(void) const { return gram_inv_; };

  /**
   * Return the Gram matrix
   * \return Gram matrix
   */
  const Eigen::MatrixXd& gram(void) const { return gram_; };
  
protected:
  void setParameterVectorAll(const Eigen::VectorXd& values);
  
private:
  Eigen::MatrixXd train_inputs_;
  Eigen::MatrixXd train_targets_;
  Eigen::MatrixXd gram_;
  double maximum_covariance_;
  Eigen::VectorXd sigmas_;

  // Cached variables, computed only in the constructor
  Eigen::VectorXd gram_inv_targets_;
  Eigen::MatrixXd gram_inv_;

  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  ModelParametersGPR(void) {};

  /** Give boost serialization access to private members. */  
  friend class boost::serialization::access;
  
  /** Serialize class data members to boost archive. 
   * \param[in] ar Boost archive
   * \param[in] version Version of the class
   * See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/tutorial.html#simplecase
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version);

};

}

#include <boost/serialization/export.hpp>
/** Register this derived class. */
BOOST_CLASS_EXPORT_KEY2(DmpBbo::ModelParametersGPR, "ModelParametersGPR")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::ModelParametersGPR,boost::serialization::object_serializable);

#endif        //  #ifndef MODELPARAMETERSGPR_H

