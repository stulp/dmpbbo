/**
 * @file   FunctionApproximatorGPR.hpp
 * @brief  FunctionApproximatorGPR class header file.
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

#ifndef _FUNCTION_APPROXIMATOR_GPR_H_
#define _FUNCTION_APPROXIMATOR_GPR_H_

#include "functionapproximators/FunctionApproximator.hpp"

/** @defgroup GPR Gaussian Process Regression (GPR)
 *  @ingroup FunctionApproximators
 */

namespace DmpBbo {
  
// Forward declarations
class MetaParametersGPR;
class ModelParametersGPR;

/** \brief GPR (Gaussian Process Regression) function approximator
 * \ingroup FunctionApproximators
 * \ingroup GPR  
 */
class FunctionApproximatorGPR : public FunctionApproximator
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
  FunctionApproximatorGPR(const MetaParametersGPR *const meta_parameters, const ModelParametersGPR *const model_parameters=NULL);  

  /** Initialize a function approximator with model parameters
   *  \param[in] model_parameters The parameters of the (previously) trained model.
   */
  FunctionApproximatorGPR(const ModelParametersGPR *const model_parameters);

	FunctionApproximator* clone(void) const;
  
	void train(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target);

	void predict(const Eigen::MatrixXd& input, Eigen::MatrixXd& output);

	void predictVariance(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& variances);
  
	inline std::string getName(void) const {
    return std::string("GPR");  
  };
  
	bool saveGridData(const Eigen::VectorXd& min, const Eigen::VectorXd& max, const Eigen::VectorXi& n_samples_per_dim, std::string directory, bool overwrite=false) const;
	
private:
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  FunctionApproximatorGPR(void) {};
   
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
BOOST_CLASS_EXPORT_KEY2(DmpBbo::FunctionApproximatorGPR, "FunctionApproximatorGPR")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::FunctionApproximatorGPR,boost::serialization::object_serializable);

#endif // _FUNCTION_APPROXIMATOR_GPR_H_


