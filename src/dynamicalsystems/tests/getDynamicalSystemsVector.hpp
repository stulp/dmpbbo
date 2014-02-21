/**
 * @file getDynamicalSystemsVector.hpp
 * @brief  Header file for function to that returns a list of DynamicalSystem objects.
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

#ifndef GETDYNAMICALSYSTEMSVECTOR_H
#define GETDYNAMICALSYSTEMSVECTOR_H

#include <vector>

namespace DmpBbo {
  
class DynamicalSystem;

void getDynamicalSystemsVector(std::vector<DynamicalSystem*>& dyn_systems);

}

#endif        //  #ifndef GETDYNAMICALSYSTEMSVECTOR_H

