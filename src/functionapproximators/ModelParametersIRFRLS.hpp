/**
 * @file   ModelParametersIRFRLS.hpp
 * @brief  ModelParametersIRFRLS class header file.
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
 
#ifndef MODELPARAMETERSIRFRLS_H
#define MODELPARAMETERSIRFRLS_H

#include <iosfwd>

#include "functionapproximators/ModelParameters.hpp"

namespace DmpBbo {

/** \brief Model parameters for the iRFRLS function approximator
 * \ingroup FunctionApproximators
 * \ingroup IRFRLS
 */
class ModelParametersIRFRLS : public ModelParameters
{
  friend class FunctionApproximatorIRFRLS;
  
public:
  /** Constructor for the model parameters of the IRFRLS function approximator.
   *  \param[in] linear_models Coefficient of the linear models (nb_cos x nb_out_dim).
   *  \param[in] cosines_periodes Matrix of periode for each cosine for each input dimension (nb_cos x nb_in_dim). 
   *  \param[in] cosines_phase Vector of periode (nb_cos).
   */
  ModelParametersIRFRLS(Eigen::VectorXd linear_models, Eigen::MatrixXd cosines_periodes, Eigen::VectorXd cosines_phase);
	
	int getExpectedInputDim(void) const;
	
	std::string toString(void) const;

  ModelParameters* clone(void) const;

  void getSelectableParameters(std::set<std::string>& selected_values_labels) const;
  void getParameterVectorMask(const std::set<std::string> selected_values_labels, Eigen::VectorXi& selected_mask) const;
  void getParameterVectorAll(Eigen::VectorXd& all_values) const;
  
  inline int getParameterVectorAllSize(void) const
  {
    return all_values_vector_size_;
  }

  UnifiedModel* toUnifiedModel(void) const;
  
  /** Return the weights of the basis functions.
   * \return weights of the basis functions.
   */
  const Eigen::VectorXd& weights(void) const { return weights_; }  

  void cosineActivations(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& cosine_activations) const;
  
protected:
  void setParameterVectorAll(const Eigen::VectorXd& values);
  
private:
  
  Eigen::VectorXd weights_;
  Eigen::MatrixXd cosines_periodes_;
  Eigen::VectorXd cosines_phase_;

  int nb_in_dim_;

  int  all_values_vector_size_;
  
public:
	/** Turn caching for the function cosineActivations() on or off.
	 * Turning this on should lead to substantial improvements in execution time if the periods and
	 * phases of the cosine functions do not change often AND you call cosineActivations with the
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
  mutable Eigen::MatrixXd cosine_activations_cached_;
  bool caching_;
  inline void clearCache(void) 
  {
    inputs_cached_.resize(0,0);
    cosine_activations_cached_.resize(0,0);
  }
  
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  ModelParametersIRFRLS(void) {};

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
BOOST_CLASS_EXPORT_KEY2(DmpBbo::ModelParametersIRFRLS, "ModelParametersIRFRLS")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::ModelParametersIRFRLS,boost::serialization::object_serializable);

#endif        //  #ifndef MODELPARAMETERSIRFRLS_H

