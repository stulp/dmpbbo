/**
 * @file   from_jsonpickle.cpp
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

#include "dynamicalsystems/from_jsonpickle.hpp"

#include "dynamicalsystems/ExponentialSystem.hpp"
#include "dynamicalsystems/SigmoidSystem.hpp"
#include "dynamicalsystems/SpringDamperSystem.hpp"
#include "dynamicalsystems/TimeSystem.hpp"

#include <iostream>
#include <nlohmann/json.hpp>

using namespace std;

namespace DmpBbo {

void DynamicalSystemFactory::from_jsonpickle(const nlohmann::json& json, DynamicalSystem*& ds) {
  
  string class_name = json.at("py/object").get<string>();
  
  if (class_name.find("ExponentialSystem") != string::npos) {
    ds = ExponentialSystem::from_jsonpickle(json);
    
  } else if (class_name.find("SigmoidSystem") != string::npos) {
    ds = SigmoidSystem::from_jsonpickle(json);
    
  } else if (class_name.find("SpringDamperSystem") != string::npos) {
    ds = SpringDamperSystem::from_jsonpickle(json);
    
  } else if (class_name.find("TimeSystem") != string::npos) {
    ds = TimeSystem::from_jsonpickle(json);
    
  } else {
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "Unknown DynamicalSystem: " << class_name << endl;
    ds = NULL;
  }
  
}

}
