/**
 * @file DmpContextualTwoStep.hpp
 * @brief  Contextual Dmp class header file.
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

#ifndef _DMP_CONTEXTUAL_TWO_STEP_H_
#define _DMP_CONTEXTUAL_TWO_STEP_H_

#include "dmp/DmpContextual.hpp"

#include <set>

namespace DmpBbo {

class FunctionApproximator;
class FunctionApproximatorLWR;

/** \defgroup Dmps Dynamic Movement Primitives
 */

/** 
 * \brief Implementation of Contextual Dynamical Movement Primitives.
 * \ingroup Dmps
 */
class DmpContextualTwoStep : public DmpContextual
{
public:
  
  /**
   *  Initialization constructor for Contextual DMPs of known dimensionality, but with unknown
   *  initial and attractor states. Initializes the DMP with default dynamical systems.
   *  \param n_dims_dmp      Dimensionality of the DMP
   *  \param function_approximators Function approximators for the forcing term
   *  \param policy_parameter_function Function approximators for the policy parameter function
   *  \param dmp_type  The type of DMP, see Dmp::DmpType    
   */
  DmpContextualTwoStep(int n_dims_dmp, std::vector<FunctionApproximator*> function_approximators, FunctionApproximator* policy_parameter_function, DmpType dmp_type=KULVICIUS_2012_JOINING);
  
  // Overloads Dmp::computeFunctionApproximatorOutput
  void computeFunctionApproximatorOutput(const Eigen::MatrixXd& phase_state, Eigen::MatrixXd& fa_output) const;

  // Overloads Dmp::train
  void  train(const std::vector<Trajectory>& trajectories, const std::vector<Eigen::MatrixXd>& task_parameters, std::string save_directory="", bool overwrite = true);
  
  /** Return whether the DMP is trained or not.
   * \return true if the DMP is trained, false otherwise 
   */
  bool isTrained(void) const;

  
private:
  std::vector<std::vector<FunctionApproximator*> > policy_parameter_function_;

protected:
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. See \ref sec_boost_serialization_ugliness
   */
   DmpContextualTwoStep(void) {}; 
   
private:
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
BOOST_CLASS_EXPORT_KEY2(DmpBbo::DmpContextualTwoStep, "DmpContextualTwoStep")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::DmpContextualTwoStep,boost::serialization::object_serializable);

#endif