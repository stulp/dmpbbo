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
 

#include "bbo/DistributionGaussian.hpp"

#include "bbo/serialization.hpp"

// If the SAVE_XML flag is defined, an xml is saved and loaded. If is is not, it is only loaded.
//#define SAVE_XML 

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
  std::string filename("/tmp/distribution.xml");

#ifdef SAVE_XML
    // Save to xml file (if SAVE_XML flag is defined)
    int dim = 2;
    VectorXd mean(dim);
    mean << 0.0, 1.0;
    MatrixXd covar(dim,dim);
    covar << 3.0, 0.5, 0.5, 1.0;
    MatrixXd covar_diag(dim,dim);
    covar_diag << 3.0, 0.0, 0.0, 1.0;
    
    DistributionGaussian* obj_save = new DistributionGaussian(mean, covar); 
    cout << "Saving DistributionGaussian to " << filename << endl;
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
    DistributionGaussian* obj_load;
    ia >> BOOST_SERIALIZATION_NVP(obj_load);
    ifs.close();

    
    // Output toString and xml output
    //cout << obj_load->toString() << endl;
    boost::archive::xml_oarchive oa2(std::cout);
    oa2 << BOOST_SERIALIZATION_NVP(obj_load);

    /*
  
  
  Updater* updaters[3];
  
#ifdef SAVE_XML


  double eliteness = 10;
  string weighting_method = "PI-BB";
  updaters[0] = new UpdaterMean(eliteness,weighting_method);

  double covar_decay_factor = 0.9;
  updaters[1] = new UpdaterCovarDecay(eliteness,covar_decay_factor,weighting_method);
  
  VectorXd base_level = VectorXd::Constant(n_dims,0.01);
  bool diag_only = true;
  double learning_rate = 1.0;
  updaters[2] = new UpdaterCovarAdaptation(eliteness,weighting_method,base_level,diag_only,learning_rate);  


#endif // SAVE_XML
  
  
  
  for (unsigned int dd=0; dd<3; dd++)
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
  */
}

