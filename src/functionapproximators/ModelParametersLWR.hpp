/**
 * @file   ModelParametersLWR.hpp
 * @brief  ModelParametersLWR class header file.
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
 
#ifndef MODELPARAMETERSLWR_H
#define MODELPARAMETERSLWR_H

#include <iosfwd>
#include <vector>
#include <set>
#include <eigen3/Eigen/Core>


#include <nlohmann/json_fwd.hpp>


namespace DmpBbo {

/** \brief Model parameters for the Locally Weighted Regression (LWR) function approximator
 * \ingroup FunctionApproximators
 * \ingroup LWR
 */
class ModelParametersLWR
{
  friend class FunctionApproximatorLWR;
  
public:
  /** Constructor for the model parameters of the LWPR function approximator.
   *  \param[in] centers Centers of the basis functions
   *  \param[in] widths  Widths of the basis functions. 
   *  \param[in] slopes  Slopes of the line segments. 
   *  \param[in] offsets Offsets of the line segments, i.e. the value of the line segment at its intersection with the y-axis.
   * \param[in] asymmetric_kernels Whether to use asymmetric kernels or not, cf MetaParametersLWR::asymmetric_kernels()
   * \param[in] lines_pivot_at_max_activation Whether line models should pivot at x=0 (false), or at the center of the kernel (x=x_c)
   */
  ModelParametersLWR(const Eigen::MatrixXd& centers, const Eigen::MatrixXd& widths, const Eigen::MatrixXd& slopes, const Eigen::MatrixXd& offsets, bool asymmetric_kernels=false, bool lines_pivot_at_max_activation=false);
  
  std::string toString(void) const;

	
  int getExpectedInputDim(void) const  {
    return centers_.cols();
  };
    	
  /** Get the unnormalized kernel activations for given inputs
   * \param[in] inputs The input data (size: n_samples X n_dims)
   * \param[out] kernel_activations The kernel activations, computed for each of the samples in the input data (size: n_samples X n_basis_functions)
   */
  void unnormalizedKernelActivations(const Eigen::Ref<const Eigen::MatrixXd>& inputs, Eigen::MatrixXd& kernel_activations) const;

  /** Get the normalized kernel activations for given inputs
   * \param[in] inputs The input data (size: n_samples X n_dims)
   * \param[out] kernel_activations The normalized kernel activations, computed for each of the sampels in the input data (size: n_samples X n_basis_functions)
   */
  void kernelActivations(const Eigen::Ref<const Eigen::MatrixXd>& inputs, Eigen::MatrixXd& kernel_activations) const;
  
  /** Get the output of each linear model (unweighted) for the given inputs.
   * \param[in] inputs The inputs for which to compute the output of the lines models (size: n_samples X  n_input_dims)
   * \param[out] lines The output of the linear models (size: n_samples X n_basis_functions) 
   *
   * If "lines" is passed as a Matrix of correct size (n_samples X n_basis_functions), this function
   * will not allocate any memory, and is real-time.
   */
  void getLines(const Eigen::Ref<const Eigen::MatrixXd>& inputs, Eigen::MatrixXd& lines) const;
  
  void setParameterVectorModifierPrivate(std::string modifier, bool new_value);
  
  /** Set whether the offsets should be adapted so that the line segments pivot around the mode of
   * the basis function, rather than the intersection with the y-axis.
   * \param[in] lines_pivot_at_max_activation Whether to pivot around the mode or not.
   *
   */
  void set_lines_pivot_at_max_activation(bool lines_pivot_at_max_activation);

  /** Whether to return slopes as angles or slopes in ModelParametersLWR::getParameterVectorAll()
   * \param[in] slopes_as_angles Whether to return as slopes (true) or angles (false)
   * \todo Implement and document
   */
  void set_slopes_as_angles(bool slopes_as_angles);
  
  /** Get the number of basis functions in this model.
   * \return The number of basis functions.
   */
  inline unsigned int getNumberOfBasisFunctions() const
  {
    return centers_.rows();
  }
  
  static ModelParametersLWR* from_jsonpickle(const nlohmann::json& json);
  
  // https://github.com/nlohmann/json/issues/1324
  friend void to_json(nlohmann::json& j, const ModelParametersLWR& m);
  //friend void from_json(const nlohmann::json& j, ModelParametersRBFN& m);
  
private:
  Eigen::MatrixXd centers_; // n_centers X n_dims
  Eigen::MatrixXd widths_;  // n_centers X n_dims
  Eigen::MatrixXd slopes_;  // n_centers X n_dims
  Eigen::VectorXd offsets_; // n_centers X 1

  bool asymmetric_kernels_; // should be const
  bool lines_pivot_at_max_activation_;
  bool slopes_as_angles_;
  
  /** Default constructor to faciliate deserialization */
  ModelParametersLWR(void) {};
};

}

#endif        //  #ifndef MODELPARAMETERSLWR_H

