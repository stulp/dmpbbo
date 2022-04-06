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
 
#include "dmp/Dmp.hpp"

#include <iostream>
#include <nlohmann/json.hpp>

using namespace std;

namespace DmpBbo {

void from_jsonpickle(const nlohmann::json& json, Dmp*& dmp) {
  
  string class_name = json.at("py/object").get<string>();
  
  if (class_name.find("Dmp") != string::npos) {
    dmp = Dmp::from_jsonpickle(json);
    
  } else {
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "Unknown Dmp: " << class_name << endl;
    dmp = NULL;
  }
  
}

}
