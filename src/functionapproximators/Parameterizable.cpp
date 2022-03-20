/**
 * @file   Parameterizable.cpp
 * @brief  Parameterizable class source file.
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

#include "functionapproximators/Parameterizable.hpp"

#include <iostream>
#include <limits>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

namespace DmpBbo {


void Parameterizable::setParameterVectorModifier(std::string modifier, bool new_value)
{
  if (parameter_vector_all_initial_.size()>0)
  {
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "Warning: you can only set a ParameterVectorModifier if the intial state has not yet been determined." << endl;
    return;
  }
  setParameterVectorModifierPrivate(modifier, new_value);
}

}
