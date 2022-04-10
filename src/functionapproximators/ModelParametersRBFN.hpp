/**
 * @file   ModelParametersRBFN.hpp
 * @brief  ModelParametersRBFN class header file.
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
 
#ifndef MODELPARAMETERSRBFN_H
#define MODELPARAMETERSRBFN_H

#include <iosfwd>
#include <vector>

#include <eigen3/Eigen/Core>


#include <nlohmann/json_fwd.hpp>

namespace DmpBbo {


/** \brief Model parameters for the Radial Basis Function Network (RBFN) function approximator
 * \ingroup FunctionApproximators
 * \ingroup RBFN
 */
class ModelParametersRBFN
{
  friend class FunctionApproximatorRBFN;
  
public:
  /** Constructor for the model parameters of the LWPR function approximator.
   *  \param[in] centers Centers of the basis functions
   *  \param[in] widths  Widths of the basis functions. 
   *  \param[in] weights Weight of each basis function
   */
  ModelParametersRBFN(const Eigen::MatrixXd& centers, const Eigen::MatrixXd& widths, const Eigen::MatrixXd& weights);
  
  std::string toString(void) const;
	
  int getExpectedInputDim(void) const  {
    return centers_.cols();
  };
  
  /** Get the number of basis functions in this model.
   * \return The number of basis functions.
   */
  inline unsigned int getNumberOfBasisFunctions() const
  {
    return centers_.rows();
  }
  
  /** Get the kernel activations for given inputs
   * \param[in] inputs The input data (size: n_samples X n_dims)
   * \param[out] kernel_activations The kernel activations, computed for each of the samples in the input data (size: n_samples X n_basis_functions)
   */
  void kernelActivations(const Eigen::Ref<const Eigen::MatrixXd>& inputs, Eigen::MatrixXd& kernel_activations) const;
  
  /** Return the weights of the basis functions.
   * \return weights of the basis functions.
   */
  const Eigen::VectorXd& weights(void) const { return weights_; }  
  
  /** Return the weights of the basis functions.
   * \param[out] weights of the basis functions.
   */
  inline void weights(Eigen::VectorXd& weights) const { weights=weights_; }  

  // https://github.com/nlohmann/json/issues/1324
  static ModelParametersRBFN* from_jsonpickle(const nlohmann::json& json);
  
  friend void to_json(nlohmann::json& j, const ModelParametersRBFN& m);
  //friend void from_json(const nlohmann::json& j, ModelParametersRBFN& m);
  
private:
  Eigen::MatrixXd centers_; // n_centers X n_dims
  Eigen::MatrixXd widths_;  // n_centers X n_dims
  Eigen::VectorXd weights_; //         1 X n_dims

  /** Default constructor.*/
  ModelParametersRBFN(void) {};
  
  
};

}

#endif        //  #ifndef MODELPARAMETERSRBFN_H

