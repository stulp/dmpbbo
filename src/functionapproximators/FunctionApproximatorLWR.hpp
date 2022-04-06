/**
 * @file   FunctionApproximatorLWR.hpp
 * @brief  FunctionApproximatorLWR class header file.
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

#ifndef _FUNCTION_APPROXIMATOR_LWR_H_
#define _FUNCTION_APPROXIMATOR_LWR_H_

#include "functionapproximators/FunctionApproximator.hpp"

#include <nlohmann/json_fwd.hpp>

/** @defgroup LWR Locally Weighted Regression (LWR)
 *  @ingroup FunctionApproximators
 */

namespace DmpBbo {
  
// Forward declarations
class MetaParametersLWR;
class ModelParametersLWR;

/** \brief LWR (Locally Weighted Regression) function approximator
 * \ingroup FunctionApproximators
 * \ingroup LWR  
 */
class FunctionApproximatorLWR : public FunctionApproximator
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
  FunctionApproximatorLWR(const MetaParametersLWR *const meta_parameters, const ModelParametersLWR *const model_parameters=NULL);  

  /** Initialize a function approximator with model parameters
   *  \param[in] model_parameters The parameters of the (previously) trained model.
   */
  FunctionApproximatorLWR(const ModelParametersLWR *const model_parameters);

  FunctionApproximatorLWR(int expected_input_dim, const Eigen::VectorXi& n_basis_functions_per_dim, double intersection_height=0.5, double regularization=0.0, bool asymmetric_kernels=false);
  
	FunctionApproximator* clone(void) const;
  
  static FunctionApproximatorLWR* from_jsonpickle(nlohmann::json json);
  // https://github.com/nlohmann/json/issues/1324
  friend void to_json(nlohmann::json& j, const FunctionApproximatorLWR& m);
  //friend void from_json(const nlohmann::json& j, FunctionApproximatorLWR& m);
  
	void train(const Eigen::Ref<const Eigen::MatrixXd>& inputs, const Eigen::Ref<const Eigen::MatrixXd>& targets);

  /** Query the function approximator to make a prediction
   *  \param[in]  inputs   Input values of the query
   *  \param[out] outputs  Predicted output values
   *
   * \remark This method should be const. But third party functions which is called in this function
   * have not always been implemented as const (Examples: LWPRObject::predict).
   * Therefore, this function cannot be const.
   *
   * This function is realtime if inputs.rows()==1 (i.e. only one input sample is provided), and the
   * memory for outputs is preallocated.
   */
	void predict(const Eigen::Ref<const Eigen::MatrixXd>& inputs, Eigen::MatrixXd& outputs);
  
	inline std::string getName(void) const {
    return std::string("LWR");  
  };
  
	bool saveGridData(const Eigen::VectorXd& min, const Eigen::VectorXd& max, const Eigen::VectorXi& n_samples_per_dim, std::string directory, bool overwrite=false) const;

private:  
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  FunctionApproximatorLWR(void) {};
   
  /** Give boost serialization access to private members. */  
  friend class boost::serialization::access;
  
  /** Serialize class data members to boost archive. 
   * \param[in] ar Boost archive
   * \param[in] version Version of the class
   * See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/tutorial.html#simplecase
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(FunctionApproximator);
  }
  
  void preallocateMemory(int n_basis_functions);
  
  /** Preallocated memory for one time step, required to make the predict() function real-time. */
  mutable Eigen::MatrixXd lines_one_prealloc_;
  
  /** Preallocated memory for one time step, required to make the predict() function real-time. */
  mutable Eigen::MatrixXd activations_one_prealloc_;
  
  /** Preallocated memory to make things more efficient. */
  mutable Eigen::MatrixXd lines_prealloc_;
  
  /** Preallocated memory to make things more efficient. */
  mutable Eigen::MatrixXd activations_prealloc_;
  
};

}

#endif // _FUNCTION_APPROXIMATOR_LWR_H_


