/**
 * @file   from_jsonpickle.hpp
 * @author Freek Stulp
 *
 * This file is part of DmpBbo, a set of libraries and programs for the 
 * black-box optimization of dynamical movement primitives.
 * Copyright (C) 2022 Freek Stulp
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

#ifndef _FUNCTION_APPROXIMATOR_FACTORY_H_
#define _FUNCTION_APPROXIMATOR_FACTORY_H_

#include <nlohmann/json_fwd.hpp>


/** @ingroup FunctionApproximators
 */

 
namespace DmpBbo {

// Forward declaration
class FunctionApproximator;

class FunctionApproximatorFactory {

public:

static void from_jsonpickle(const nlohmann::json& json, FunctionApproximator*& fa);

/** Initialize a function approximator from its name.
 \param[in] name Name of the function approximator, e.g. "LWR"
 \param[in] n_input_dims Dimensionality of the input data
 \return A pointer to an initialized function approximator.
 */
static FunctionApproximator* getFunctionApproximatorByName(std::string name, int n_input_dims=1);

/**
LWR : n_basis_functions (int), intersection (double)
RBFN : n_basis_functions (int), intersection (double)
LWPR: w_gen, w_prune, update_D, init_alpha, penalty, init_d    
GMR: n_basis_functions (int)
RRRFF n_basis_functions (int), regularization (double), gamma (double)
GPR: maximum_covariance (double), length (double)
 */
static FunctionApproximator* getFunctionApproximatorFromArgs(int n_args, char* args[], int n_input_dims=1);

};

}

#endif // _FUNCTION_APPROXIMATOR_FACTORY_H_