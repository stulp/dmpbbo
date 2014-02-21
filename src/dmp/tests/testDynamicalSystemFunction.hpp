/**
 * @file testDynamicalSystemFunction.hpp
 * @brief  Header file for function to test a Dynamical System.
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

#ifndef _TEST_DYNAMICAL_SYSTEM_FUNCTION_H_
#define _TEST_DYNAMICAL_SYSTEM_FUNCTION_H_

#include <string>

namespace DmpBbo {
  
// Forward declaration
class DynamicalSystem;

/** Function to test a dynamical system, i.e. do numerical integration, compute the analytical
 *  solution, save results to file, etc.
 *
 * \param[in] dyn_sys The dynamical system to test
 * \param[in] dt      Integration constant
 * \param[in] T       Number of integration steps
 * \param[in] output_analytical_filename Name of output file for analytical data (optional)
 * \param[in] output_step_filename Name of output file for numerical integration data (optional)
 */
void testDynamicalSystemFunction(DynamicalSystem* dyn_sys, double dt, int T, std::string directory="");

}

#endif


