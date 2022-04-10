/**
 * @file   FunctionApproximatorRBFN.hpp
 * @brief  FunctionApproximatorRBFN class header file.
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

#ifndef _FUNCTION_APPROXIMATOR_RBFN_H_
#define _FUNCTION_APPROXIMATOR_RBFN_H_

#include "functionapproximators/FunctionApproximator.hpp"

#include <nlohmann/json_fwd.hpp>


/** @defgroup RBFN Radial Basis Function Network (RBFN)
 *  @ingroup FunctionApproximators
 */

namespace DmpBbo {
  
// Forward declarations
class MetaParametersRBFN;
class ModelParametersRBFN;

/** \brief RBFN (Radial Basis Function Network) function approximator
 * \ingroup FunctionApproximators
 * \ingroup RBFN  
 */
class FunctionApproximatorRBFN : public FunctionApproximator
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
  FunctionApproximatorRBFN(const MetaParametersRBFN *const meta_parameters, const ModelParametersRBFN *const model_parameters=NULL);  

  /** Initialize a function approximator with model parameters
   *  \param[in] model_parameters The parameters of the (previously) trained model.
   */
  FunctionApproximatorRBFN(const ModelParametersRBFN *const model_parameters);

  static FunctionApproximatorRBFN* from_jsonpickle(nlohmann::json json);
  // https://github.com/nlohmann/json/issues/1324
  friend void to_json(nlohmann::json& j, const FunctionApproximatorRBFN& m);
  //friend void from_json(const nlohmann::json& j, FunctionApproximatorRBFN& m);
  
	FunctionApproximator* clone(void) const;
  
	void predict(const Eigen::Ref<const Eigen::MatrixXd>& inputs, Eigen::MatrixXd& output);
  
	/** Preallocate memory to make certain functions real-time.
	 * \param[in] n_basis_functions Number of basis functions in the RBFN.
	 */
  void preallocateMemory(int n_basis_functions);
  
	inline std::string getName(void) const {
    return std::string("RBFN");  
  };

	bool saveGridData(const Eigen::VectorXd& min, const Eigen::VectorXd& max, const Eigen::VectorXi& n_samples_per_dim, std::string directory, bool overwrite=false) const;
	
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  FunctionApproximatorRBFN(void) {};
   
private:  
  /** Preallocated memory to make things realtime and more efficient. */
  mutable Eigen::VectorXd weights_prealloc_;
  
  /** Preallocated memory for one time step, required to make the predict() function real-time. */
  mutable Eigen::MatrixXd activations_one_prealloc_;
  
  /** Preallocated memory to make things more efficient. */
  mutable Eigen::MatrixXd activations_prealloc_;

};

}

#endif // _FUNCTION_APPROXIMATOR_RBFN_H_


