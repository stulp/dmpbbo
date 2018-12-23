/**
 * \file demoDmpExtendedDimensions.cpp
 * \author Freek Stulp
 * \brief  Demonstrates how to initialize, train and integrate a Dmp.
 *
 * \ingroup Demos
 * \ingroup Dmps
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
#include "dmp/DmpExtendedDimensions.hpp"
#include "dmp/Trajectory.hpp"

#include "dynamicalsystems/DynamicalSystem.hpp"
#include "dynamicalsystems/ExponentialSystem.hpp"
#include "dynamicalsystems/SigmoidSystem.hpp"
#include "dynamicalsystems/TimeSystem.hpp"
#include "dynamicalsystems/SpringDamperSystem.hpp"

#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/ModelParametersLWR.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"

#include <iostream>
#include <fstream>


using namespace std;
using namespace Eigen;
using namespace DmpBbo;

/** Get a demonstration trajectory.
 * \param[in] ts The time steps at which to sample
 * \return a Demonstration trajectory
 */
Trajectory getDemoTrajectory(const VectorXd& ts);

/** Main function
 * \param[in] n_args Number of arguments
 * \param[in] args Arguments themselves
 * \return Success of exection. 0 if successful.
 */
int main(int n_args, char** args)
{
  string save_directory;
  if (n_args!=2) 
  {
    cerr << "Usage: " << args[0] << "<directory>" << endl;
    return -1;
  }
  save_directory = string(args[1]);

  // GENERATE A TRAJECTORY 
  double tau = 0.5;
  int n_time_steps = 51;
  VectorXd ts = VectorXd::LinSpaced(n_time_steps,0,tau); // Time steps
  Trajectory trajectory = getDemoTrajectory(ts); // getDemoTrajectory() is implemented below main()
  int n_dims = trajectory.dim();
  
  // MAKE THE FUNCTION APPROXIMATORS
  
  // Initialize some meta parameters for training LWR function approximator
  int n_basis_functions = 25;
  int input_dim = 1;
  double intersection = 0.5;
  MetaParametersLWR* meta_parameters = new MetaParametersLWR(input_dim,n_basis_functions,intersection);      
  FunctionApproximatorLWR* fa_lwr = new FunctionApproximatorLWR(meta_parameters);  
  
  // Clone the function approximator for each dimension of the DMP
  vector<FunctionApproximator*> function_approximators(n_dims);    
  for (int dd=0; dd<n_dims; dd++)
    function_approximators[dd] = fa_lwr->clone();
  
  // CONSTRUCT AND TRAIN THE DMP
  
  cout << "** Initialize DMP." << endl;
  // Initialize the DMP
  Dmp::DmpType dmp_type = Dmp::KULVICIUS_2012_JOINING;
  //dmp_type = Dmp::IJSPEERT_2002_MOVEMENT;
  Dmp* dmp_tmp = new Dmp(n_dims, function_approximators, dmp_type);

  cout << "** Initialize DmpExtendedDimensions." << endl;
  int n_ext_dims = trajectory.dim_misc();
  // Clone the function approximator for each external dimension of the DMP
  vector<FunctionApproximator*> function_approximators_ext_dims(n_ext_dims);    
  for (int dd=0; dd<n_ext_dims; dd++)
    function_approximators_ext_dims[dd] = fa_lwr->clone();
  
  DmpExtendedDimensions* dmp_ext_dims = new DmpExtendedDimensions(dmp_tmp,function_approximators_ext_dims);

  cout << "** Train DmpExtendedDimensions." << endl;
  // And train it. Passing the save_directory will make sure the results are saved to file.
  bool overwrite = true;
  dmp_ext_dims->train(trajectory,save_directory,overwrite);

  
  // INTEGRATE DMP TO GET REPRODUCED TRAJECTORY
  
  cout << "** Integrate DMP analytically." << endl;
  Trajectory traj_reproduced;
  tau = 0.9;
  n_time_steps = 91;
  ts = VectorXd::LinSpaced(n_time_steps,0,tau); // Time steps
  dmp_ext_dims->analyticalSolution(ts,traj_reproduced);

  // Integrate again, but this time get more information
  MatrixXd xs_ana, xds_ana, forcing_terms_ana, fa_output_ana, fa_extended_output;
  dmp_ext_dims->analyticalSolution(ts,xs_ana,xds_ana,forcing_terms_ana,fa_output_ana,fa_extended_output);

  
  // WRITE THINGS TO FILE
  trajectory.saveToFile(save_directory,"demonstration_traj.txt",overwrite);
  traj_reproduced.saveToFile(save_directory,"reproduced_traj.txt",overwrite);
    
  MatrixXd output_ana(ts.size(),1+xs_ana.cols()+xds_ana.cols());
  output_ana << ts, xs_ana, xds_ana;
  saveMatrix(save_directory,"reproduced_ts_xs_xds.txt",output_ana,overwrite);
  saveMatrix(save_directory,"reproduced_forcing_terms.txt",forcing_terms_ana,overwrite);
  saveMatrix(save_directory,"reproduced_fa_output.txt",fa_output_ana,overwrite);
  saveMatrix(save_directory,"reproduced_fa_extended.txt",fa_extended_output,overwrite);


  // INTEGRATE STEP BY STEP
  cout << "** Integrate DMP step-by-step." << endl;
  VectorXd x(dmp_ext_dims->dim(),1);
  VectorXd xd(dmp_ext_dims->dim(),1);
  VectorXd x_updated(dmp_ext_dims->dim(),1);
  VectorXd ext_dims(dmp_ext_dims->dim_extended(),1);

  MatrixXd xs_step(n_time_steps,x.size());
  MatrixXd xds_step(n_time_steps,xd.size());
  MatrixXd ext_dim_all(n_time_steps,ext_dims.size());
  
  cout << std::setprecision(3) << std::fixed << std::showpos;
  double dt = ts[1];
  dmp_ext_dims->integrateStart(x,xd,ext_dims);
  xs_step.row(0) = x;
  xds_step.row(0) = xd;
  ext_dim_all.row(0) = ext_dims;
  for (int t=1; t<n_time_steps; t++)
  {
    dmp_ext_dims->integrateStep(dt,x,x_updated,xd,ext_dims); 
    x = x_updated;
    xs_step.row(t) = x;
    xds_step.row(t) = xd;
    ext_dim_all.row(t) = ext_dims;
    if (save_directory.empty())
    {
      // Not writing to file, output on cout instead.
      //cout << x.transpose() << " | " << xd.transpose() << endl;
    }
  } 

  MatrixXd output_step(ts.size(),1+xs_ana.cols()+xds_ana.cols()+ext_dim_all.cols());
  output_step << ts, xs_step, xds_step, ext_dim_all;
  saveMatrix(save_directory,"reproduced_step_ts_xs_xds_ext.txt",output_step,overwrite);

  
  delete meta_parameters;
  delete fa_lwr;
  delete dmp_ext_dims;

  return 0;
}

Trajectory getDemoTrajectory(const VectorXd& ts)
{
  bool use_viapoint_traj= false;
  Trajectory trajectory;
  int n_dims=0;
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
    trajectory =  Trajectory::generateMinJerkTrajectory(ts, y_first, y_last);
  }
  
  // Generated trajectory, now generate extended dimensions in width
  MatrixXd misc(ts.rows(),n_dims);
  for (int ii=0; ii<misc.rows(); ii++)
  {
    misc(ii,0) = (1.0*ii)/misc.rows();
    if (n_dims>1)
      misc(ii,1) = sin((8.0*ii)/misc.rows());
  }
  
  trajectory.set_misc(misc);
  
  return trajectory;
}