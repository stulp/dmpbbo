/**
 * @file   getFunctionApproximatorByName.hpp
 * @brief  getFunctionApproximatorByName header file.
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

/** \defgroup FunctionApproximators Function Approximators
 */

#ifndef _GETFUNCTIONAPPROXIMATORBYNAME_H_
#define _GETFUNCTIONAPPROXIMATORBYNAME_H_

#include <string>

namespace DmpBbo {

class FunctionApproximator;

/** Initialize a function approximator from its name.
 \param[in] name Name of the function approximator, e.g. "LWR"
 \param[in] n_input_dims Dimensionality of the input data
 \return A pointer to an initialized function approximator.
 */
FunctionApproximator* getFunctionApproximatorByName(std::string name, int n_input_dims=1);

/**
LWR : n_basis_functions (int), intersection (double)
RBFN : n_basis_functions (int), intersection (double)
LWPR: w_gen, w_prune, update_D, init_alpha, penalty, init_d    
GMR: n_basis_functions (int)
IRFRLS n_basis_functions (int), lambda (double), gamma (double)
GPR: maximum_covariance (double), length (double)
 */
FunctionApproximator* getFunctionApproximatorFromArgs(int n_args, char* args[], int n_input_dims=1);

}

#endif
