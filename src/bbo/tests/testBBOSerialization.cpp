/**
 * @file testBBOSerialization.cpp
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
 

#include "dynamicalsystems/DynamicalSystem.hpp"

#include "dynamicalsystems/serialization.hpp"

// If the SAVE_XML flag is defined, an xml is saved and loaded. If is is not, it is only loaded.
#define SAVE_XML 

#ifdef SAVE_XML
// No need to include this header if we read the objects from file. 
#include "getDynamicalSystemsVector.hpp"
#endif // SAVE_XML


#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>


#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;
using namespace DmpBbo;  

int main(int n_args, char** args)
{
  unsigned int n_systems;
  
#ifdef SAVE_XML
  // When saving to xml, initialize some objects to save
  vector<DynamicalSystem*> dyn_systems;
  getDynamicalSystemsVector(dyn_systems);
  n_systems = dyn_systems.size();
#else
  // Hack; we know "dyn_systems" contains 5 objects 
  n_systems = 5; 
#endif // SAVE_XML
  

  for (unsigned int dd=0; dd<n_systems; dd++)
  {
    // Prepare filename
    std::string filename("/tmp/dyn_sys");
    filename += to_string(dd)+".xml";

    cout << "___________________________________________" << endl;
    
    
#ifdef SAVE_XML
    // Save to xml file (if SAVE_XML flag is defined)
    DynamicalSystem* obj_save = dyn_systems[dd]; 
    cout << "Saving " << obj_save->toString() << " to " << filename << endl;
    std::ofstream ofs(filename);
    boost::archive::xml_oarchive oa(ofs);
    oa << BOOST_SERIALIZATION_NVP(obj_save);
    ofs.close();
#else
    cout << "(NOT saving object to " << filename << ")" << endl;
#endif // SAVE_XML


    // Load file from xml file
    cout << "Loading from " << filename << endl;
    std::ifstream ifs(filename);
    boost::archive::xml_iarchive ia(ifs);
    DynamicalSystem* obj_load;
    ia >> BOOST_SERIALIZATION_NVP(obj_load);
    ifs.close();

    
    // Output toString and xml output
    cout << obj_load->toString() << endl;
    boost::archive::xml_oarchive oa2(std::cout);
    oa2 << BOOST_SERIALIZATION_NVP(obj_load);
    
  }
}

