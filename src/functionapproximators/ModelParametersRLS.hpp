/**
 * @file   ModelParametersRLS.hpp
 * @brief  ModelParametersRLS class header file.
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
 
#ifndef MODELPARAMETERSRLS_H
#define MODELPARAMETERSRLS_H

#include "functionapproximators/ModelParameters.hpp"

#include <iosfwd>
#include <vector>

#include <eigen3/Eigen/Core>

namespace DmpBbo {

/** \brief Model parameters for (Regularized) Least Squares 
 * \ingroup FunctionApproximators
 * \ingroup RLS
 */
class ModelParametersRLS : public ModelParameters
{
  friend class FunctionApproximatorRLS;
  
public:
  /** Constructor for the model parameters of the LWPR function approximator.
   *  \param[in] slopes  Slopes of the line. 
   */
  ModelParametersRLS(const Eigen::VectorXd& slopes);
  
  /** Constructor for the model parameters of the LWPR function approximator.
   *  \param[in] slopes  Slopes of the line. 
   *  \param[in] offset Offset of the line segment, i.e. the value of the line segment at its intersection with the y-axis.
   */
  ModelParametersRLS(const Eigen::VectorXd& slopes, double offset);
  
  std::string toString(void) const;
  
	ModelParameters* clone(void) const;
	
  int getExpectedInputDim(void) const  {
    return slopes_.size();
  };
  
  /** Get the output of each linear model (unweighted) for the given inputs.
   * \param[in] inputs The inputs for which to compute the output of the lines models (size: n_samples X  n_input_dims)
   * \param[out] lines The output of the linear models (size: n_samples X 1) 
   *
   * If "lines" is passed as a Matrix of correct size (n_samples X 1), this function
   * will not allocate any memory, and is real-time.
   */
  void getLines(const Eigen::Ref<const Eigen::MatrixXd>& inputs, Eigen::VectorXd& lines) const;
  
  void setParameterVectorModifierPrivate(std::string modifier, bool new_value);
  
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
  Eigen::VectorXd slopes_;  // size: n_dims
  double offset_;
  double use_offset_;
  
  int  all_values_vector_size_;
	
private:
  
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  ModelParametersRLS(void) {};

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
BOOST_CLASS_EXPORT_KEY2(DmpBbo::ModelParametersRLS, "ModelParametersRLS")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::ModelParametersRLS,boost::serialization::object_serializable);

#endif        //  #ifndef MODELPARAMETERSRLS_H

