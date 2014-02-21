/**
 * @file testDynamicalSystemSerialization.cpp
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

#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;
using namespace DmpBbo;  

int main(int n_args, char** args)
{
  
  vector<DynamicalSystem*> dyn_systems;
  getDynamicalSystemsVector(dyn_systems);
  
  for (unsigned int dd=0; dd<dyn_systems.size(); dd++)
  {
    DynamicalSystem* cur_dyn_system = dyn_systems[dd]; 
    
    // create and open a character archive for output
    std::string filename("/tmp/dyn_sys");
    filename += to_string(dd)+".xml";
    
    
    std::ofstream ofs(filename);
    boost::archive::xml_oarchive oa(ofs);
    //oa << BOOST_SERIALIZATION_NVP(dyn_sys);
    oa << boost::serialization::make_nvp("dyn_sys",cur_dyn_system);
    ofs.close();
  
    std::ifstream ifs(filename);
    boost::archive::xml_iarchive ia(ifs);
    DynamicalSystem* dyn_sys_out;
    ia >> BOOST_SERIALIZATION_NVP(dyn_sys_out);
    ifs.close();
    
    cout << "___________________________________________" << endl;
    cout << "  filename=" << filename << endl;
    cout << *cur_dyn_system << endl;
    cout << *dyn_sys_out << endl;
    
  }
}

