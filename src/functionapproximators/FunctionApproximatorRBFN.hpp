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
  
/** \brief RBFN (Radial Basis Function Network) function approximator
 * \ingroup FunctionApproximators
 * \ingroup RBFN  
 */
class FunctionApproximatorRBFN : public FunctionApproximator
{
public:
  

  /** Constructor for the model parameters of the function approximator.
   *  \param[in] centers Centers of the basis functions
   *  \param[in] widths  Widths of the basis functions. 
   *  \param[in] weights Weight of each basis function
   */
  FunctionApproximatorRBFN(const Eigen::MatrixXd& centers, const Eigen::MatrixXd& widths, const Eigen::MatrixXd& weights);

  static FunctionApproximatorRBFN* from_jsonpickle(nlohmann::json json);
  // https://github.com/nlohmann/json/issues/1324
  friend void to_json(nlohmann::json& j, const FunctionApproximatorRBFN& m);
  //friend void from_json(const nlohmann::json& j, FunctionApproximatorRBFN& m);
  
	void predict(const Eigen::Ref<const Eigen::MatrixXd>& inputs, Eigen::MatrixXd& output) const;
  
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  FunctionApproximatorRBFN(void) {};

  ~FunctionApproximatorRBFN(void) {};

  
	std::string toString(void) const;
	
private:  
  
  int n_basis_functions_;
  Eigen::MatrixXd centers_; // n_basis_functions_ X n_dims
  Eigen::MatrixXd widths_;  // n_basis_functions_ X n_dims
  Eigen::VectorXd weights_; //                  1 X n_dims
  
  /** Preallocated memory for one time step, required to make the predict() function real-time. */
  mutable Eigen::MatrixXd activations_one_prealloc_;
  
};

}

#endif // _FUNCTION_APPROXIMATOR_RBFN_H_


