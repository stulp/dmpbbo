/**
 * @file   UnifiedModel.hpp
 * @brief  UnifiedModel class header file.
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
 
#ifndef UNIFIEDMODEL_H
#define UNIFIEDMODEL_H

#include "functionapproximators/Parameterizable.hpp"

#include <iosfwd>
#include <vector>

#include <eigen3/Eigen/Core>

namespace DmpBbo {

/** \page page_unified_model Unified Model
 * 
Whilst coding this library and numerous discussion with Olivier Sigaud, it became apparent that the latent function representations of all the function approximators in this library all use the same generic model. Each specific model (i.e. as used in GPR, GMR, LWR, etc.) is a special case of the Unified Model. We discuss this in a forthcoming paper titled: "Many Regression Algorithms, One Unified Model - A Review, Freek Stulp and Olivier Sigaud", which you should be able to find in an on-line search.
 *
 */

/** \brief The unified model, which can be used to represent the model of all other function approximators.
 *
 * Also see the page on the \ref page_unified_model
 * \ingroup FunctionApproximators
 */
class UnifiedModel : public Parameterizable
{
  friend class FunctionApproximatorLWR;
  
public:
  /** Constructor for the unified model parameters. This version is used by for example RBFN and GPR. 
   *  \param[in] centers Centers of the basis functions (n_bfs X n_input_dims)
   *  \param[in] widths  Widths of the basis functions (n_bfs X n_input_dims)
   *  \param[in] weights Weights of each basis function (n_bfs X 1)
   * \param[in] normalized_basis_functions Whether to use normalized basis functions
   * \param[in] lines_pivot_at_max_activation Whether line models should pivot at x=0 (false), or at the center of the kernel (x=x_c)
   */
  UnifiedModel(const Eigen::MatrixXd& centers, const Eigen::MatrixXd& widths, const Eigen::VectorXd& weights, bool normalized_basis_functions, bool lines_pivot_at_max_activation=false);

  /** Constructor for the unified model parameters. This version is used by for example LWR and LWPR 
   *  \param[in] centers Centers of the basis functions (n_bfs X n_input_dims)
   *  \param[in] widths  Widths of the basis functions (n_bfs X n_input_dims)
   *  \param[in] slopes  Slopes of the line segments (n_bfs X n_input_dims)
   *  \param[in] offsets Offsets of the line segments, i.e. the value of the line segment at its intersection with the y-axis (n_bfs X 1)
   * \param[in] normalized_basis_functions Whether to use normalized basis functions or not
   * \param[in] lines_pivot_at_max_activation Whether line models should pivot at x=0 (false), or at the center of the kernel (x=x_c)
   */
  UnifiedModel(const Eigen::MatrixXd& centers, const Eigen::MatrixXd& widths, const Eigen::MatrixXd& slopes, const Eigen::VectorXd& offsets, bool normalized_basis_functions, bool lines_pivot_at_max_activation=false);

  /** Constructor for the unified model parameters. This version is used by for example IRFRLS 
   *  \param[in] angular_frequencies Angular frequency for each dimension and each cosine basis function  (n_bfs X n_input_dims)
   *  \param[in] phases  Phase of each cosine basis function (n_bfs X 1)
   *  \param[in] weights Weights of each basis function (n_bfs X 1)
   */
  UnifiedModel(const Eigen::MatrixXd& angular_frequencies, const Eigen::VectorXd& phases, const Eigen::VectorXd& weights);

  /** Constructor for the unified model parameters. This version is used by for example GMR. 
   *  \param[in] centers Centers of the basis functions (n_bfs X n_input_dims)
   *  \param[in] widths  Widths of the basis functions (n_bfs X n_input_dims)
   *  \param[in] slopes  Slopes of the line segments (n_bfs X n_input_dims)
   *  \param[in] offsets Offsets of the line segments, i.e. the value of the line segment at its intersection with the y-axis (n_bfs X 1)
   *  \param[in] priors Prior of each basis function (n_bfs X 1)
   * \param[in] normalized_basis_functions Whether to use normalized basis functions or not
   * \param[in] lines_pivot_at_max_activation Whether line models should pivot at x=0 (false), or at the center of the kernel (x=x_c)
   */
  UnifiedModel(
    const std::vector<Eigen::VectorXd>& centers, // n_centers X n_dims
    const std::vector<Eigen::MatrixXd>& widths,  // n_centers X n_dims X n_dims
    const std::vector<Eigen::VectorXd>& slopes, // n_centers X n_dims
    const std::vector<double>& offsets,          // n_centers X 1
    const std::vector<double>& priors,           // n_centers X 1              
    bool normalized_basis_functions=false, 
    bool lines_pivot_at_max_activation=false);
  
  
  /** Returns a string representation of the object.
   * \return A string representation of the object.
   */
  std::string toString(void) const;
  
  /** Return a pointer to a deep copy of the object.
   *  \return Pointer to a deep copy
   */
	UnifiedModel* clone(void) const;
	
  /** The expected dimensionality of the input data.
   * \return Expected dimensionality of the input data
   */
  int getExpectedInputDim(void) const  {
    if (centers_.size()>0)
      return centers_[0].size();
    else
      return 0;
  };
  
  /** Get the kernel activations for given inputs
   * \param[in] inputs The input data (size: n_samples X n_dims)
   * \param[out] kernel_activations The kernel activations, computed for each of the samples in the input data (size: n_samples X n_basis_functions)
   */
  void kernelActivations(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& kernel_activations) const;
  
  /** Get the output of each linear model (unweighted) for the given inputs.
   * \param[in] inputs The inputs for which to compute the output of the lines models (size: n_samples X  n_input_dims)
   * \param[out] lines The output of the linear models (size: n_samples X n_output_dim) 
   */
  void getLines(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& lines) const;
  
  /** Compute the sum of the locally weighted lines. 
   * \param[in] inputs The inputs for which to compute the output (size: n_samples X  n_input_dims)
   * \param[out] output The weighted linear models (size: n_samples X n_output_dim) 
   *
   */
  void evaluate(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& output) const;
  
  void setParameterVectorModifierPrivate(std::string modifier, bool new_value);
  
  /** Set whether the offsets should be adapted so that the line segments pivot around the mode of
   * the basis function, rather than the intersection with the y-axis.
   * \param[in] lines_pivot_at_max_activation Whether to pivot around the mode or not.
   *
   */
  void set_lines_pivot_at_max_activation(bool lines_pivot_at_max_activation);

  /** Whether to return slopes as angles or slopes in UnifiedModel::getParameterVectorAll()
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
  
  /** Generate a grid of inputs, and output the response of the basis functions and line segments
   * for these inputs.
   * This function is not pure virtual, because this might not make sense for every model parameters
   * class.
   *
   * \param[in] min Minimum values for the grid (one for each dimension)
   * \param[in] max Maximum values for the grid (one for each dimension)
   * \param[in] n_samples_per_dim Number of samples in the grid along each dimension
   * \param[in] directory Directory to which to save the results to.
   * \param[in] overwrite Whether to overwrite existing files. true=do overwrite, false=don't overwrite and give a warning.
   * \return Whether saving the data was successful.
   */
	bool saveGridData(const Eigen::VectorXd& min, const Eigen::VectorXd& max, const Eigen::VectorXi& n_samples_per_dim, std::string directory, bool overwrite=false) const;

protected:
  void setParameterVectorAll(const Eigen::VectorXd& values);
  
private:
  std::vector<Eigen::VectorXd> centers_; // n_centers X n_dims
  std::vector<Eigen::MatrixXd> covars_;  // n_centers X n_dims X n_dims
  std::vector<Eigen::VectorXd> slopes_;  // n_centers X n_dims
  std::vector<double> offsets_;          // n_centers X 1
  std::vector<double> priors_;           // n_centers X 1

  bool cosine_basis_functions_;
  bool normalized_basis_functions_;
  bool lines_pivot_at_max_activation_;
  bool slopes_as_angles_;
  int  all_values_vector_size_;
  void initializeAllValuesVectorSize(void);

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
  UnifiedModel(void) {};

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
BOOST_CLASS_EXPORT_KEY2(DmpBbo::UnifiedModel, "UnifiedModel")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::UnifiedModel,boost::serialization::object_serializable);

#endif        //  #ifndef UNIFIEDMODEL_H

