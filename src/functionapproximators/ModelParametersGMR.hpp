/**
 * @file   ModelParametersGMR.hpp
 * @brief  ModelParametersGMR class header file.
 * @author Freek Stulp, Thibaut Munzer
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
 
#ifndef MODELPARAMETERSGMR_H
#define MODELPARAMETERSGMR_H

#include "functionapproximators/ModelParameters.hpp"

#include <iosfwd>
#include <vector>

namespace DmpBbo {

/** \brief Model parameters for the GMR function approximator
 * \ingroup FunctionApproximators
 * \ingroup GMR
 */
class ModelParametersGMR : public ModelParameters
{
  friend class FunctionApproximatorGMR;
  
public:
  /** Constructor for the model parameters of the GMR function approximator.
   *  \param[in] centers A list (std::vector) of nb_gaussian center vector (nb_in_dim)
   *  \param[in] priors A list (std::vector) of nb_gaussian prior
   *  \param[in] slopes A list (std::vector) of nb_gaussian associated slope matrix (nb_out_dim x nb_in_dim)
   *  \param[in] biases A list (std::vector) of nb_gaussian associated bias vector (nb_out_dim)
   *  \param[in] inverseCovarsL A list (std::vector) of nb_gaussian matrix. Each matrix is the inverse of the L part of the LLT decomposition of the covariance matrix (nb_in_dim x nb_in_dim)
   */
  ModelParametersGMR(std::vector<double> priors, std::vector<Eigen::VectorXd> mu_xs,
    std::vector<Eigen::VectorXd> mu_ys, std::vector<Eigen::MatrixXd> sigma_xs,
    std::vector<Eigen::MatrixXd> sigma_ys, std::vector<Eigen::MatrixXd> sigma_x_ys);

	int getExpectedInputDim(void) const;
	
	std::string toString(void) const;

  ModelParameters* clone(void) const;

  /** @todo Determine which parameters should be modifiable in GMR. */
  void getSelectableParameters(std::set<std::string>& selected_values_labels) const;
  void getParameterVectorMask(const std::set<std::string> selected_values_labels, Eigen::VectorXi& selected_mask) const;
  void getParameterVectorAll(Eigen::VectorXd& all_values) const;
  inline int getParameterVectorAllSize(void) const
  {
    return all_values_vector_size_;
  }
  
  static bool saveGMM(std::string directory, const std::vector<Eigen::VectorXd>& centers, const std::vector<Eigen::MatrixXd>& covars, int iter=-1);
  
	bool saveGridData(const Eigen::VectorXd& min, const Eigen::VectorXd& max, const Eigen::VectorXi& n_samples_per_dim, std::string directory, bool overwrite=false) const;

protected:
  void setParameterVectorAll(const Eigen::VectorXd& values);
  
private:
  std::vector<double> priors_;

  std::vector<Eigen::VectorXd> means_x_;
  std::vector<Eigen::VectorXd> means_y_;

  std::vector<Eigen::MatrixXd> covars_x_;
  std::vector<Eigen::MatrixXd> covars_y_;
  std::vector<Eigen::MatrixXd> covars_y_x_;

  /** This is covars_x_.inverse(). Since we used it often, we cache it here. */
  std::vector<Eigen::MatrixXd> covars_x_inv_;

  int  all_values_vector_size_;
  
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  ModelParametersGMR(void) {};
  
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
BOOST_CLASS_EXPORT_KEY2(DmpBbo::ModelParametersGMR, "ModelParametersGMR")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::ModelParametersGMR,boost::serialization::object_serializable);

#endif        //  #ifndef MODELPARAMETERSGMR_H

