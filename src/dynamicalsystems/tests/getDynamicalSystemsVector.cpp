/**
 * @file getDynamicalSystemsVector.cpp
 * @brief  Source file for function to that returns a list of DynamicalSystem objects.
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

#include "getDynamicalSystemsVector.hpp"

#include "dynamicalsystems/DynamicalSystem.hpp"
#include "dynamicalsystems/ExponentialSystem.hpp"
#include "dynamicalsystems/SigmoidSystem.hpp"
#include "dynamicalsystems/TimeSystem.hpp"
#include "dynamicalsystems/SpringDamperSystem.hpp"

#include <vector>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

namespace DmpBbo {
  
void getDynamicalSystemsVector(vector<DynamicalSystem*>& dyn_systems)
{
  // ExponentialSystem
  double tau = 0.6; // Time constant
  VectorXd initial_state(2);   initial_state   << 0.5, 1.0; 
  VectorXd attractor_state(2); attractor_state << 0.8, 0.1; 
  double alpha = 6.0; // Decay factor
  dyn_systems.push_back(new ExponentialSystem(tau, initial_state, attractor_state, alpha));
  
  // TimeSystem
  dyn_systems.push_back(new TimeSystem(tau));

  // TimeSystem (but counting down instead of up)
  bool count_down = true;
  dyn_systems.push_back(new TimeSystem(tau,count_down));

  // SigmoidSystem
  double max_rate = -20;
  double inflection_point = tau*0.8;
  dyn_systems.push_back(new SigmoidSystem(tau, initial_state, max_rate, inflection_point));
  
  // SpringDamperSystem
  alpha = 12.0;
  dyn_systems.push_back(new SpringDamperSystem(tau, initial_state, attractor_state, alpha));

}

}
