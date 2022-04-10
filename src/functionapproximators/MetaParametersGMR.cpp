/**
 * @file   MetaParametersGMR.cpp
 * @brief  MetaParametersGMR class source file.
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

#include "functionapproximators/MetaParametersGMR.hpp"


#include <iostream>

using namespace std;

namespace DmpBbo {

MetaParametersGMR::MetaParametersGMR(int expected_input_dim, int number_of_gaussians) 
	:
    MetaParameters(expected_input_dim),
    number_of_gaussians_(number_of_gaussians)
{
}

MetaParametersGMR* MetaParametersGMR::clone(void) const 
{
  return new MetaParametersGMR(getExpectedInputDim(),number_of_gaussians_);
}

string MetaParametersGMR::toString(void) const 
{
  return string("MetaParametersGMR");
}

}
