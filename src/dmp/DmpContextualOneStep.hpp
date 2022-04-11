/**
 * @file DmpContextualOneStep.hpp
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

#ifndef _DMP_CONTEXTUAL_ONE_STEP_H_
#define _DMP_CONTEXTUAL_ONE_STEP_H_

#include "dmp/DmpContextual.hpp"

#include <set>


namespace DmpBbo {

class FunctionApproximator;

/** 
 * \brief Implementation of Contextual Dynamical Movement Primitives.
 * \ingroup DmpContextual
 */
class DmpContextualOneStep : public DmpContextual
{
public:
  
  /**
   *  Initialization constructor for Contextual DMPs of known dimensionality, but with unknown
   *  initial and attractor states. Initializes the DMP with default dynamical systems.
   *  \param n_dims_dmp      Dimensionality of the DMP
   *  \param function_approximators Function approximators for the forcing term
   *  \param dmp_type  The type of DMP, see Dmp::DmpType    
   */
  DmpContextualOneStep(int n_dims_dmp, std::vector<FunctionApproximator*> function_approximators,
    DmpType dmp_type="KULVICIUS_2012_JOINING");
    
  // Overrides DmpContextual::computeFunctionApproximatorOutput
  void computeFunctionApproximatorOutput(
    const Eigen::Ref<const Eigen::MatrixXd>& phase_state, Eigen::MatrixXd& fa_output) const;

  // Overloads Dmp::train
  void  train(const std::vector<Trajectory>& trajectories, const std::vector<Eigen::MatrixXd>& task_parameters, std::string save_directory="", bool overwrite=false);
  
};

}

#endif