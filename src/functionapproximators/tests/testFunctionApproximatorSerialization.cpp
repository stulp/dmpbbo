/**
 * \file testFunctionApproximatorSerialization.cpp
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

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <time.h>
#include <boost/filesystem.hpp>

#include "functionapproximators/ModelParameters.hpp"
#include "functionapproximators/MetaParameters.hpp"
#include "functionapproximators/FunctionApproximator.hpp"

#include "getFunctionApproximatorsVector.hpp"
#include "../demos/targetFunction.hpp"

#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

int main(int n_args, char** args)
{
  vector<string> fa_names;
  if (n_args>1)
  {
    for (int aa=1; aa<n_args; aa++)
      fa_names.push_back(string(args[aa]));
  }
  else
  {
    fa_names.push_back("LWR");
    fa_names.push_back("IRFRLS");
    fa_names.push_back("LWPR");
    fa_names.push_back("GMR");
  }
        
  
  for (int n_input_dims=1; n_input_dims<=2; n_input_dims++)
  {
    cout << "___________________________________________________________________" << endl;
    cout << "Using " << n_input_dims << "-D data.   " << endl;
    
    VectorXi n_samples_per_dim = VectorXi::Constant(1,10);
    if (n_input_dims==2) 
      n_samples_per_dim = VectorXi::Constant(2,25);
    
    MatrixXd inputs, targets, outputs;
    targetFunction(n_samples_per_dim,inputs,targets);
    
    
    for (unsigned int i_name=0; i_name<fa_names.size(); i_name++)
    {
      // GMR on 2D too slow
      if (fa_names[i_name].compare("GMR")==0)
        if (n_input_dims==2)
          continue;
        
      FunctionApproximator* cur_obj = getFunctionApproximatorByName(fa_names[i_name],n_input_dims);
      if (cur_obj==NULL)
        continue;
      
      for (int trained=0; trained<=1; trained++)
      {
        cout << "______________________________________________" << endl;
        if (trained==1)
        {
          cout << "Training function approximator..." << endl;
          cur_obj->train(inputs,targets);
        }
        
        stringstream stream;
        FunctionApproximator* cur_obj_out = NULL;
        if (cur_obj==NULL)
          continue;
        stream << *cur_obj;
        
        // create and open a character archive for output
        std::string filename("/tmp/fa_obj_");
        filename += to_string(i_name)+".xml";
        std::ofstream ofs(filename);
        boost::archive::xml_oarchive oa(ofs);
        oa << boost::serialization::make_nvp("Name",cur_obj);
        ofs.close();
      
        std::ifstream ifs(filename);
        boost::archive::xml_iarchive ia(ifs);
        ia >> BOOST_SERIALIZATION_NVP(cur_obj_out);
        ifs.close();
        
        cout << "___________________________________________" << endl;
        cout << filename << endl;
        cout << *cur_obj << endl;
        cout << *cur_obj_out << endl;
        
      }
      
    }
  
  }
  
  return 0;
}


