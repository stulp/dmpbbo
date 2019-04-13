/**
 * \file trainDmpTrainFromTrajectoryFile.cpp
 * \author Freek Stulp
 * \brief  Demonstrates how to train a Dmp with a trajectory in a txt file.
 *
 * \ingroup Demos
 * \ingroup Dmps // zzz
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

#include "dmp/Dmp.hpp"
#include "dmp/Trajectory.hpp"

#include "dynamicalsystems/DynamicalSystem.hpp"
#include "dynamicalsystems/ExponentialSystem.hpp"
#include "dynamicalsystems/SigmoidSystem.hpp"
#include "dynamicalsystems/TimeSystem.hpp"
#include "dynamicalsystems/SpringDamperSystem.hpp"

#include "functionapproximators/FunctionApproximatorRBFN.hpp"
#include "functionapproximators/MetaParametersRBFN.hpp"
#include "functionapproximators/ModelParametersRBFN.hpp"
#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/ModelParametersLWR.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"

#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>


using namespace std;
using namespace Eigen;
using namespace DmpBbo;

void help(char* binary_name)
{
  cout << "Usage: " << binary_name << " <input trajectory (txt)> <output directory> [output dmp (xml)] [n_basis_functions]" << endl;
  cout << "Default for dmp: 'dmp.xml' " << endl;
  cout << "Default for n_basis_functions: 5 " << endl;
}

/** Main function
 * \param[in] n_args Number of arguments
 * \param[in] args Arguments themselves
 * \return Success of exection. 0 if successful.
 */
int main(int n_args, char** args)
{
  
  string input_txt_file;
  string directory;
  string output_xml_file("dmp.xml");
  int n_basis_functions = 5;
  
  if (n_args<3)
  {
    help(args[0]);
    return -1;
  }
  
  if (string(args[1]).compare("--help")==0)
  {
    help(args[0]);
    return 0;
  }

  input_txt_file = string(args[1]);
  directory = string(args[2]);
  if (n_args>3)
    output_xml_file = string(args[3]);
  if (n_args>4)
    n_basis_functions = atoi(args[4]);
    
  
  cout << "Reading trajectory from TXT file: " << input_txt_file << endl;
  Trajectory trajectory = Trajectory::readFromFile(input_txt_file);
  if (trajectory.length()==0)
  {
    cerr << "The TXT file " << input_txt_file << " could not be found. Aborting." << endl << endl;
    help(args[0]);
    return -1;
  }

  //double tau = trajectory.duration();
  //int n_time_steps = trajectory.length();
  VectorXd ts = trajectory.ts(); // Time steps
  int n_dims = trajectory.dim();

  
  // Initialize some meta parameters for training RBFN function approximator
  int input_dim = 1;
  double intersection = 0.7;
  MetaParametersRBFN* meta_parameters = new MetaParametersRBFN(input_dim,n_basis_functions,intersection);      
  FunctionApproximatorRBFN* fa_lwr = new FunctionApproximatorRBFN(meta_parameters);  
  
  // Set the parameters to optimize
  set<string> parameters_to_optimize;
  parameters_to_optimize.insert("weights");
  
  // Clone the function approximator for each dimension of the DMP
  vector<FunctionApproximator*> function_approximators(n_dims);    
  for (int dd=0; dd<n_dims; dd++)
    function_approximators[dd] = fa_lwr->clone();
  
  // Initialize the DMP
  Dmp* dmp = new Dmp(n_dims, function_approximators, Dmp::KULVICIUS_2012_JOINING);

  cout << "Training Dmp... (n_basis_functions=" << n_basis_functions << ")" << endl;
  bool overwrite = true;
  dmp->train(trajectory,directory+"/train",overwrite);

  // Make directory if it doesn't already exist
  if (!boost::filesystem::exists(directory))
  {
    if (!boost::filesystem::create_directories(directory))
    {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "Couldn't make directory file '" << directory << "'." << endl;
      return false;
    }
  }

  cout << "Writing trained Dmp to XML file: " << directory << "/" << output_xml_file << endl;
  std::ofstream ofs(directory+"/"+output_xml_file);
  boost::archive::xml_oarchive oa(ofs);
  oa << boost::serialization::make_nvp("dmp",dmp);
  ofs.close();
  
  // Set which parameters to optimize, and save the initial vector to file
  dmp->setSelectedParameters(parameters_to_optimize);
  Eigen::VectorXd parameter_vector;
  dmp->getParameterVectorSelected(parameter_vector);
  overwrite = true;
  cout << "Writing initial parameter vector to file : " << directory << "/parameter_vector_initial.txt" << endl;
  saveMatrix(directory,"parameter_vector_initial.txt",parameter_vector,overwrite);
  
    
  delete meta_parameters;
  delete fa_lwr;
  delete dmp;

  return 0;
}
