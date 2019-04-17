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
#include "dmp/serialization.hpp"

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

#include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

void help(char* binary_name)
{
  cout << "Usage: " << binary_name << " <dmp filename.xml> <input/output directory> " << endl;
}

/** Main function
 * \param[in] n_args Number of arguments
 * \param[in] args Arguments themselves
 * \return Success of exection. 0 if successful.
 */
int main(int n_args, char** args)
{
  
  if (n_args!=3)
  {
    help(args[0]);
    return -1;
  }
  
  if (string(args[1]).compare("--help")==0)
  {
    help(args[0]);
    return 0;
  }

  string dmp_filename = string(args[1]);
  string directory = string(args[2]);
  
  cout << directory << endl;
  MatrixXd samples;
  if (!loadMatrix(directory+"samples.txt", samples)) return -1;
  cout << "  samples=" << samples << endl;
  
  std::ifstream ifs(dmp_filename);
  boost::archive::xml_iarchive ia(ifs);
  Dmp* dmp;
  ia >> BOOST_SERIALIZATION_NVP(dmp);
  ifs.close();
  
  cout << *dmp << endl;

  // Integrate DMP longer than the tau with which it was trained
  double integration_time = 1.5*dmp->tau();
  double frequency_Hz = 100.0;
  int n_time_steps = floor(frequency_Hz*integration_time);
  VectorXd ts = VectorXd::LinSpaced(n_time_steps,0,integration_time); // Time steps
  
  // Save trajectory without perturbation
  Trajectory traj;
  dmp->analyticalSolution(ts,traj);
  bool overwrite = true;
  traj.saveToFile(directory, "traj_unperturbed.txt", overwrite);

  // Save trajectories with perturbed samples 
  VectorXd cur_sample;
  for (int i_sample=0; i_sample<samples.rows(); i_sample++)
  {
    cur_sample = samples.row(i_sample);
    dmp->setParameterVectorSelected(cur_sample);
    dmp->analyticalSolution(ts,traj);
    
    stringstream stream;
    stream << "traj_sample" << setw(5) << setfill('0') << i_sample+1 << ".txt";
    traj.saveToFile(directory, stream.str(), overwrite);
    
  }
  
  /*

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
  */
  delete dmp;
  return 0;
}
