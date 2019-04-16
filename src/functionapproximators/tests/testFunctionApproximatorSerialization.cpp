/**
 * @file testMetaParametersSerialization.cpp
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
 

#include "functionapproximators/ModelParameters.hpp"
#include "functionapproximators/MetaParameters.hpp"
#include "functionapproximators/FunctionApproximator.hpp"

#include "functionapproximators/serialization.hpp"

// If the SAVE_XML flag is defined, an xml is saved and loaded. If is is not, it is only loaded.
#define SAVE_XML 

#ifdef SAVE_XML
// No need to include this header if we read the objects from file. 
#include "functionapproximators/getFunctionApproximatorByName.hpp"
#endif // SAVE_XML


#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>


#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;
using namespace DmpBbo;  

void targetFunction(Eigen::VectorXi n_samples_per_dim, Eigen::MatrixXd& inputs, Eigen::MatrixXd& targets);

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
    fa_names.push_back("RBFN");
    fa_names.push_back("LWR");
    fa_names.push_back("RRRFF");
    //fa_names.push_back("LWPR");
    fa_names.push_back("GPR");
    fa_names.push_back("GMR");
  }

  for (unsigned int i_name=0; i_name<fa_names.size(); i_name++)
  {
    // Prepare filename
    std::string filename("/tmp/function_approximator_");
    filename += fa_names[i_name]+".xml";
  
    cout << "___________________________________________" << endl;
    
#ifdef SAVE_XML

    // When saving to xml, initialize the object
    FunctionApproximator* obj_save_fa = getFunctionApproximatorByName(fa_names[i_name]);
    if (obj_save_fa==NULL)
      continue;
    
    // Train the FunctionApproximator
    VectorXi n_samples_per_dim = VectorXi::Constant(1,10);
    MatrixXd inputs, targets, outputs;
    targetFunction(n_samples_per_dim,inputs,targets);
    cout << "Training function approximator..." << endl;
    obj_save_fa->train(inputs,targets);
    
    //const MetaParameters* obj_save = obj_save_fa->getMetaParameters();
    //const ModelParameters* obj_save = obj_save_fa->getModelParameters();
    FunctionApproximator* obj_save = obj_save_fa;
    
    // Save to xml file 
    cout << "Saving " << fa_names[i_name] << " to " << filename << endl;
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
    //MetaParameters* obj_load;
    //ModelParameters* obj_load;
    FunctionApproximator* obj_load;
    ia >> BOOST_SERIALIZATION_NVP(obj_load);
    ifs.close();
  
    
    // Output xml
    boost::archive::xml_oarchive oa2(std::cout);
    oa2 << BOOST_SERIALIZATION_NVP(obj_load);
  
  }
}


/** Target function.
 *  \param[in] n_samples_per_dim The number of samples along each dimension. 
 *  \param[in] inputs The input vector
 *  \param[out] targets The target values for that input vector.
 */
void targetFunction(Eigen::VectorXi n_samples_per_dim, Eigen::MatrixXd& inputs, Eigen::MatrixXd& targets)
{
  int n_dims = n_samples_per_dim.size();
  if (n_dims==1)
  {
    // 1D Function:  y =  3*e^(-x) * sin(2*x^2);
    inputs = Eigen::VectorXd::LinSpaced(n_samples_per_dim[0], 0.0, 2.0);
    targets = 3*(-inputs.col(0)).array().exp()*(2*inputs.col(0).array().pow(2)).sin();

  }
  else
  {
    // 2D Function, similar to the example and graph here: http://www.mathworks.com/help/matlab/visualize/mapping-data-to-transparency-alpha-data.html
    int n_samples = n_samples_per_dim[0]*n_samples_per_dim[1];
    inputs = Eigen::MatrixXd::Zero(n_samples, n_dims);
    Eigen::VectorXd x1 = Eigen::VectorXd::LinSpaced(n_samples_per_dim[0], -2.0, 2.0);
    Eigen::VectorXd x2 = Eigen::VectorXd::LinSpaced(n_samples_per_dim[1], -2.0, 2.0);
    for (int ii=0; ii<x1.size(); ii++)
    {
      for (int jj=0; jj<x2.size(); jj++)
      {
        inputs(ii*x2.size()+jj,0) = x1[ii];
        inputs(ii*x2.size()+jj,1) = x2[jj];
      }
    }
    targets = 2.5*inputs.col(0).array()*exp(-inputs.col(0).array().pow(2) - inputs.col(1).array().pow(2));
    
  }
}

