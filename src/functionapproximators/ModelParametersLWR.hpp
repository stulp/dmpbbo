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

#include "functionapproximators/ModelParameters.hpp"

#include <iosfwd>
#include <vector>

#include <eigen3/Eigen/Core>

namespace DmpBbo {

/** \brief Model parameters for the Locally Weighted Regression (LWR) function approximator
 * \ingroup FunctionApproximators
 * \ingroup LWR
 */
class ModelParametersLWR : public ModelParameters
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
  
	ModelParameters* clone(void) const;
	
  int getExpectedInputDim(void) const  {
    return centers_.cols();
  };
    	
  /** Get the unnormalized kernel activations for given inputs
   * \param[in] inputs The input data (size: n_samples X n_dims)
   * \param[out] kernel_activations The kernel activations, computed for each of the samples in the input data (size: n_samples X n_basis_functions)
   */
  void unnormalizedKernelActivations(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& kernel_activations) const;

  /** Get the normalized kernel activations for given inputs
   * \param[in] inputs The input data (size: n_samples X n_dims)
   * \param[out] kernel_activations The normalized kernel activations, computed for each of the sampels in the input data (size: n_samples X n_basis_functions)
   */
  void kernelActivations(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& kernel_activations) const;
  
  /** Get the output of each linear model (unweighted) for the given inputs.
   * \param[in] inputs The inputs for which to compute the output of the lines models (size: n_samples X  n_input_dims)
   * \param[out] lines The output of the linear models (size: n_samples X n_output_dim) 
   */
  void getLines(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& lines) const;
  
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
  
  void getSelectableParameters(std::set<std::string>& selected_values_labels) const;
  void getParameterVectorMask(const std::set<std::string> selected_values_labels, Eigen::VectorXi& selected_mask) const;
  void getParameterVectorAll(Eigen::VectorXd& all_values) const;
  inline int getParameterVectorAllSize(void) const
  {
    return all_values_vector_size_;
  }
  
  UnifiedModel* toUnifiedModel(void) const;
  
protected:
  void setParameterVectorAll(const Eigen::VectorXd& values);
  
private:
  Eigen::MatrixXd centers_; // n_centers X n_dims
  Eigen::MatrixXd widths_;  // n_centers X n_dims
  Eigen::MatrixXd slopes_;  // n_centers X n_dims
  Eigen::VectorXd offsets_; // n_centers X 1

  bool asymmetric_kernels_; // should be const
  bool lines_pivot_at_max_activation_;
  bool slopes_as_angles_;
  int  all_values_vector_size_;

public:
	/** Turn caching for the function normalizedKernelActivations() on or off.
	 * Turning this on should lead to substantial improvements in execution time if the centers and
	 * widths of the kernels do not change often AND you call normalizedKernelActivations with the
	 * same inputs over and over again.
	 * \param[in] caching Whether to turn caching on or off
	 * \remarks In the constructor, caching is set to true, so by default it is on.
	 */
	inline void set_caching(bool caching)
	{
	  caching_ = caching;
	  if (!caching_) clearCache();
	}
	
private:
  
  mutable Eigen::MatrixXd inputs_cached_;
  mutable Eigen::MatrixXd kernel_activations_cached_;
  bool caching_;
  inline void clearCache(void) 
  {
    inputs_cached_.resize(0,0);
    kernel_activations_cached_.resize(0,0);
  }
  
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  ModelParametersLWR(void) {};

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
BOOST_CLASS_EXPORT_KEY2(DmpBbo::ModelParametersLWR, "ModelParametersLWR")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::ModelParametersLWR,boost::serialization::object_serializable);

#endif        //  #ifndef MODELPARAMETERSLWR_H

