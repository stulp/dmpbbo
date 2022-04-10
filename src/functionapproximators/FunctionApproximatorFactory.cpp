/**
 * @file   FunctionApproximatorFactory.cpp
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
 
#include "functionapproximators/FunctionApproximatorFactory.hpp"
#include "functionapproximators/FunctionApproximator.hpp"

#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/FunctionApproximatorRBFN.hpp"

#include <iostream>
#include <nlohmann/json.hpp>

#include <eigen3/Eigen/Core>

using namespace Eigen;
using namespace std;

namespace DmpBbo {

void FunctionApproximatorFactory::from_jsonpickle(const nlohmann::json& json, FunctionApproximator*& fa) {
  
  string class_name = json.at("py/object").get<string>();
  
  if (class_name.find("FunctionApproximatorRBFN") != string::npos) {
    fa = FunctionApproximatorRBFN::from_jsonpickle(json);
    
  } else if (class_name.find("FunctionApproximatorLWR") != string::npos) {
    fa = FunctionApproximatorLWR::from_jsonpickle(json);
    
  } else {
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "Unknown FunctionApproximator: " << class_name << endl;
    fa = NULL;
  }
  
}

}
