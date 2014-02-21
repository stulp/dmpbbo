/**
 * \file getFunctionApproximatorsVector.hpp
 * \author Freek Stulp
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

#ifndef GETFUNCTIONAPPROXIMATORSVECTOR_H
#define GETFUNCTIONAPPROXIMATORSVECTOR_H

#include <vector>
#include <string>

namespace DmpBbo {
  
class MetaParameters;
class FunctionApproximator;

MetaParameters* getMetaParametersByName(std::string name, int input_dim);
FunctionApproximator* getFunctionApproximatorByName(std::string name, int input_dim);

void getFunctionApproximatorsVector(int input_dim, std::vector<FunctionApproximator*>& function_approximators);

}

#endif        //  #ifndef GETFUNCTIONAPPROXIMATORSVECTOR_H

