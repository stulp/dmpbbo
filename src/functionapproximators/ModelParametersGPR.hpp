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
   *  \param[in] expected_input_dim         The dimensionality of the data this function approximator expects. Although this information is already contained in the 'centers_per_dim' argument, we ask the user to pass it explicitly so that various checks on the arguments may be conducted.
   *  \param[in] maximum_covariance The maximum allowable covariance of the covar function (aka sigma)
   *  \param[in] length             Length of the covariance function, i.e. sigma^2 exp(-(x-x')^2/2l^2)
   * \param[in] gram_matrix         The Gram matrix
   */
  ModelParametersGPR(int expected_input_dim, double maximum_covariance, double length, MatrixXd gram_matrix);
   
  std::string toString(void) const;
  
	ModelParameters* clone(void) const;
	
  int getExpectedInputDim(void) const  {
    return centers_.cols();
  };
  
  
  void getSelectableParameters(std::set<std::string>& selected_values_labels) const;
  void getParameterVectorMask(const std::set<std::string> selected_values_labels, Eigen::VectorXi& selected_mask) const;
  void getParameterVectorAll(Eigen::VectorXd& all_values) const;
  inline int getParameterVectorAllSize(void) const
  {
    return 0;
  }
  
	bool saveGridData(const Eigen::VectorXd& min, const Eigen::VectorXd& max, const Eigen::VectorXi& n_samples_per_dim, std::string directory, bool overwrite=false) const;

protected:
  void setParameterVectorAll(const Eigen::VectorXd& values);
  
private:
  double maximum_covariance_;
  double length_;
  double gram_matrix_;

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

