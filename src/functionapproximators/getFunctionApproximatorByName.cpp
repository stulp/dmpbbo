/**
 * @file   getFunctionApproximatorByName.cpp
 * @brief  getFunctionApproximatorByName class source file.
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

#include "functionapproximators/getFunctionApproximatorByName.hpp"

#include "functionapproximators/FunctionApproximatorFactory.hpp"

#include <eigen3/Eigen/Core>

using namespace Eigen;
using namespace std;

namespace DmpBbo {

FunctionApproximator* getFunctionApproximatorByName(std::string name, int n_input_dims)
{
  return FunctionApproximatorFactory::getFunctionApproximatorByName(name,n_input_dims);
}

FunctionApproximator* getFunctionApproximatorFromArgs(int n_args, char* args[], int n_input_dims)
{
  return FunctionApproximatorFactory::getFunctionApproximatorFromArgs(n_args,args,n_input_dims);
}

}
