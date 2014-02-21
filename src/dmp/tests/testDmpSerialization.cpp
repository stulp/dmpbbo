/**
 * \file testDmpSerialization.cpp
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

#include "dmp/Dmp.hpp"
#include "dmp/Trajectory.hpp"

#include "dynamicalsystems/DynamicalSystem.hpp"
#include "dynamicalsystems/ExponentialSystem.hpp"
#include "dynamicalsystems/SigmoidSystem.hpp"
#include "dynamicalsystems/TimeSystem.hpp"
#include "dynamicalsystems/SpringDamperSystem.hpp"

#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/ModelParametersLWR.hpp"

#include <iostream>
#include <fstream>


using namespace std;
using namespace Eigen;
using namespace DmpBbo;

int main(int n_args, char** args)
{

  // Generate a trajectory 
  double tau = 0.5;
  int n_time_steps = 51;
  VectorXd ts = VectorXd::LinSpaced(n_time_steps,0,tau); // Time steps
  int n_dims;
  Trajectory trajectory;
  
  bool use_viapoint_traj= false;
  if (use_viapoint_traj)
  {
    n_dims = 1;
    VectorXd y_first = VectorXd::Zero(n_dims);
    VectorXd y_last  = VectorXd::Ones(n_dims);
    double viapoint_time = 0.25;
    double viapoint_location = 0.5;
  
    VectorXd y_yd_ydd_viapoint = VectorXd::Zero(3*n_dims);
    y_yd_ydd_viapoint.segment(0*n_dims,n_dims).fill(viapoint_location); // y         
    trajectory = Trajectory::generatePolynomialTrajectoryThroughViapoint(ts,y_first,y_yd_ydd_viapoint,viapoint_time,y_last); 
  }
  else
  {
    n_dims = 2;
    VectorXd y_first = VectorXd::LinSpaced(n_dims,0.0,0.7); // Initial state
    VectorXd y_last  = VectorXd::LinSpaced(n_dims,0.4,0.5); // Final state
    trajectory = Trajectory::generateMinJerkTrajectory(ts, y_first, y_last);
  }
  
  
  
  // Initialize some meta parameters for training LWR function approximator
  int n_basis_functions = 3;
  int input_dim = 1;
  double intersection = 0.56;
  MetaParametersLWR* meta_parameters = new MetaParametersLWR(input_dim,n_basis_functions,intersection);      
  FunctionApproximatorLWR* fa_lwr = new FunctionApproximatorLWR(meta_parameters);  
  
  // Clone the function approximator for each dimension of the DMP
  vector<FunctionApproximator*> function_approximators(n_dims);    
  for (int dd=0; dd<n_dims; dd++)
    function_approximators[dd] = fa_lwr->clone();
  
  // Initialize the DMP
  Dmp* dmp = new Dmp(n_dims, function_approximators, Dmp::KULVICIUS_2012_JOINING);

  for (int trained=0; trained<=1; trained++)
  {
    cout << "______________________________________________" << endl;
    if (trained==1)
    {
      cout << "Training Dmp..." << endl;
      dmp->train(trajectory);
    }
    // create and open a character archive for output
    std::string filename("/tmp/dmp_");
    filename += to_string(trained)+".xml";
    
    
    std::ofstream ofs(filename);
    boost::archive::xml_oarchive oa(ofs);
    oa << boost::serialization::make_nvp("dmp",dmp);
    ofs.close();
  
    std::ifstream ifs(filename);
    boost::archive::xml_iarchive ia(ifs);
    Dmp* dmp_out;
    ia >> BOOST_SERIALIZATION_NVP(dmp_out);
    ifs.close();
    
    cout << "___________________________________________" << endl;
    cout << "  filename=" << filename << endl;
    cout << *dmp << endl;
    cout << *dmp_out << endl;
    delete dmp_out;
  }

  delete meta_parameters;
  delete fa_lwr;
  delete dmp;

  return 0;
}
