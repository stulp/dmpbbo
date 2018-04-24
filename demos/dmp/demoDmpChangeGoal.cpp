/**
 * \file demoDmpChangeGoal.cpp
 * \author Freek Stulp
 * \brief  Demonstrates how to change the goal for a Dmp
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

  // WRITE THINGS TO FILE
  bool overwrite = true;
  trajectory.saveToFile(save_directory,"demonstration_traj.txt",overwrite);

  
  // MAKE THE FUNCTION APPROXIMATORS
  
  // Initialize some meta parameters for training LWR function approximator
  int input_dim = 1;
  //VectorXi n_basis_functions = VectorXi::Constant(input_dim,25);
  int n_basis_functions = 25;
  double intersection = 0.5;
  
  const Dmp::ForcingTermScaling scaling_vector[] = 
    {Dmp::NO_SCALING, Dmp::G_MINUS_Y0_SCALING, Dmp::AMPLITUDE_SCALING}; 
  int scaling_number;
  
  for (scaling_number=0; scaling_number<3; scaling_number++)
  {
  
    MetaParametersLWR* meta_parameters = new MetaParametersLWR(input_dim,n_basis_functions,intersection);      
    FunctionApproximatorLWR* fa_lwr = new FunctionApproximatorLWR(meta_parameters);  
    
    // Clone the function approximator for each dimension of the DMP
    vector<FunctionApproximator*> function_approximators(n_dims);    
    for (int dd=0; dd<n_dims; dd++)
      function_approximators[dd] = fa_lwr->clone();
    
    // CONSTRUCT AND TRAIN THE DMP
    
    // Initialize the DMP
    Dmp::ForcingTermScaling scaling = scaling_vector[scaling_number];
    Dmp* dmp = new Dmp(n_dims, function_approximators, Dmp::KULVICIUS_2012_JOINING, scaling);
  
    // And train it. Passing the save_directory will make sure the results are saved to file.
    string save_directory_scaling = save_directory;
    switch (scaling)
    {
    case Dmp::NO_SCALING:
      save_directory_scaling += "/NO_SCALING";
      break;
    case Dmp::G_MINUS_Y0_SCALING:
      save_directory_scaling += "/G_MINUS_Y0_SCALING";
      break;
    case Dmp::AMPLITUDE_SCALING:
      save_directory_scaling += "/AMPLITUDE_SCALING";
      break;
    }
    dmp->train(trajectory,save_directory_scaling,overwrite);
    
    
    Trajectory traj_reproduced;
    tau = 0.7;
    n_time_steps = 71;
    ts = VectorXd::LinSpaced(n_time_steps,0,tau); // Time steps
    
    // INTEGRATE DMP TO GET REPRODUCED TRAJECTORY
    for (int goal_number=0; goal_number<7; goal_number++)
    {
      VectorXd y_attr  = trajectory.final_y();
      // 0 =>  1.5
      // 1 =>  1.0
      // 2 =>  0.5
      // 3 =>  0.0
      // 4 => -0.5
      // 5 => -1.0
      // 6 => -1.5
      y_attr *= (0.5*(goal_number-3));
      
      // ANALYTICAL SOLUTION 
      dmp->set_attractor_state(y_attr);
      dmp->analyticalSolution(ts,traj_reproduced);
      string basename = string("reproduced") + to_string(goal_number);
      traj_reproduced.saveToFile(save_directory_scaling,basename+"_traj.txt",overwrite);
      
      // NUMERICAL INTEGRATION 
      VectorXd x(dmp->dim(),1);
      VectorXd xd(dmp->dim(),1);
      VectorXd x_updated(dmp->dim(),1);
      dmp->integrateStart(x,xd);

      MatrixXd xs_num(n_time_steps,x.size());
      MatrixXd xds_num(n_time_steps,xd.size());
      xs_num.row(0) = x;
      xds_num.row(0) = xd;
      
      double dt = ts[1]-ts[0];
      for (int ii=1; ii<n_time_steps; ii++)
      {
        dmp->integrateStep(dt,x,x_updated,xd); 
        x = x_updated;
        xs_num.row(ii) = x;
        xds_num.row(ii) = xd;
      }
      
      Trajectory traj_reproduced_num;
      dmp->statesAsTrajectory(ts,xs_num,xds_num,traj_reproduced_num);
      basename = string("reproduced_num") + to_string(goal_number);
      traj_reproduced_num.saveToFile(save_directory_scaling,basename+"_traj.txt",overwrite);
    
      // Integrate again, but this time get more information
      //MatrixXd xs_ana, xds_ana, forcing_terms_ana, fa_output_ana;
      //dmp->analyticalSolution(ts,xs_ana,xds_ana,forcing_terms_ana,fa_output_ana);
      //MatrixXd output_ana(ts.size(),1+xs_ana.cols()+xds_ana.cols());
      //output_ana << xs_ana, xds_ana, ts;
      //saveMatrix(save_directory_scaling,basename+"_xs_xds.txt",output_ana,overwrite);
      //saveMatrix(save_directory_scaling,basename+"_forcing_terms.txt",forcing_terms_ana,overwrite);
      //saveMatrix(save_directory_scaling,basename+"_fa_output.txt",fa_output_ana,overwrite);
    }
  
    delete meta_parameters;
    delete fa_lwr;
    delete dmp;
  }

  return 0;
}

Trajectory getDemoTrajectory(const VectorXd& ts)
{
  bool use_viapoint_traj= true;
  if (use_viapoint_traj)
  {
    int n_dims = 2;
    VectorXd y_first = VectorXd::Zero(n_dims);
    VectorXd y_last  = 0.1*VectorXd::Ones(n_dims);
    if (n_dims==2)
      y_last[1] = -0.8;
    double viapoint_time = 0.25;
    VectorXd viapoint_location = -0.5*VectorXd::Ones(n_dims);
    if (n_dims==2)
      viapoint_location[1] = -0.8;
  
    VectorXd y_yd_ydd_viapoint = VectorXd::Zero(3*n_dims);
    y_yd_ydd_viapoint.segment(0*n_dims,n_dims) = viapoint_location;
    return  Trajectory::generatePolynomialTrajectoryThroughViapoint(ts,y_first,y_yd_ydd_viapoint,viapoint_time,y_last); 
  }
  else
  {
    int n_dims = 2;
    VectorXd y_first = VectorXd::LinSpaced(n_dims,0.0,0.7); // Initial state
    VectorXd y_last  = VectorXd::LinSpaced(n_dims,0.4,0.5); // Final state
    return Trajectory::generateMinJerkTrajectory(ts, y_first, y_last);
  }
}
