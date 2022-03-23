/**
 * \file trainPerformRollout.cpp
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
  cout << "Usage: " << binary_name << " <dmp filename.xml> <output trajectory filename.txt> [dmp parameters.txt] [output dmp filename.xml]" << endl;
}

/** Main function
 * \param[in] n_args Number of arguments
 * \param[in] args Arguments themselves
 * \return Success of exection. 0 if successful.
 */
int main(int n_args, char** args)
{
  
  if (n_args<3 || n_args>5)
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
  string traj_filename = string(args[2]);
  string sample_filename = "";
  if (n_args>3)
    sample_filename = string(args[3]);
  string dmp_output_filename = "";
  if (n_args>4)
    dmp_output_filename = string(args[4]);

  cout << "C++    | Executing "; 
  for (int ii=0; ii<n_args; ii++) cout << " " << args[ii]; 
  cout << endl;
  
  // Read dmp from xml file
  cout << "C++    |     Reading dmp from file '" << dmp_filename << "'"  << endl;
  std::ifstream ifs(dmp_filename);
  boost::archive::xml_iarchive ia(ifs);
  Dmp* dmp;
  ia >> BOOST_SERIALIZATION_NVP(dmp);
  ifs.close();
  cout << "C++    |         " << *dmp << endl;

  // Read sample file, if necessary
  if (!sample_filename.empty())
  {
    cout << "C++    |     Reading sample from file '" << sample_filename << "'"  << endl;
    VectorXd sample;
    if (!loadMatrix(sample_filename, sample)) 
    {
      cerr << "C++    | WARNING: Could not read sample file. Executing default DMP instead." << endl;
    }
    else
    {
      // Set DMP parameters to sample
      dmp->setParameterVector(sample);
      // Save dmp whose parameters have been perturbed, if necessary
      if (!dmp_output_filename.empty())
      {
        cout << "C++    |     Saving dmp to file '" << dmp_output_filename << "'"  << endl;
        std::ofstream ofs(dmp_output_filename);
        boost::archive::xml_oarchive oa(ofs);
        oa << boost::serialization::make_nvp("dmp",dmp);
        ofs.close();
      }
    }
  }

  // Integrate DMP longer than the tau with which it was trained
  double integration_time = 1.5*dmp->tau();
  double frequency_Hz = 100.0;
  cout << "C++    |     Integrating dmp for " << integration_time << "s at " << (int)frequency_Hz << "Hz" << endl;
  int n_time_steps = floor(frequency_Hz*integration_time);
  VectorXd ts = VectorXd::LinSpaced(n_time_steps,0,integration_time); // Time steps
  
  // Save trajectory 
  cout << "C++    |     Saving trajectory to file '" << traj_filename << "'"  << endl;
  Trajectory trajectory;
  dmp->analyticalSolution(ts,trajectory);
  
  // Now we have the end-effector trajectory. Compute the ball trajectory.
  MatrixXd y_endeff = trajectory.ys();
  MatrixXd yd_endeff = trajectory.yds();
  MatrixXd ydd_endeff = trajectory.ydds();
  MatrixXd y_ball(n_time_steps,2); 
  MatrixXd yd_ball(n_time_steps,2);
  MatrixXd ydd_ball(n_time_steps,2);
  double dt = 1.0/frequency_Hz;
  bool ball_in_hand = true;
  for (int ii=0; ii<n_time_steps; ii++)
  {  
    if (ball_in_hand)
    {
      // If the ball is in your hand, it moves along with your hand
      y_ball.row(ii) = y_endeff.row(ii); 
      yd_ball.row(ii) = yd_endeff.row(ii); 
      ydd_ball.row(ii) = ydd_endeff.row(ii); 
      
      if (ts(ii)>0.6)
      {
        // Release the ball to throw it!
        ball_in_hand = false;
      }
    }
    else // ball_in_hand is false => ball is flying through the air
    {
        ydd_ball(ii,0) = 0.0;
        ydd_ball(ii,1) = -9.81; // Gravity
        
        // Euler integration
        yd_ball.row(ii) = yd_ball.row(ii-1) + dt*ydd_ball.row(ii);
        y_ball.row(ii) = y_ball.row(ii-1) + dt*yd_ball.row(ii);
        
        if (y_ball(ii,1)<-0.3)
        {
          // Ball hits the floor (floor is at -0.3)
          y_ball(ii,1) = -0.3;
          yd_ball.row(ii) = VectorXd::Zero(2);
          ydd_ball.row(ii) = VectorXd::Zero(2);
          
        }
    }
    
    //if x(t_i-1,BALL_IN_CUP)
    //  % If the ball is in the cup, it does not move
    //  x(t_i,BALL_X:BALL_Y) = x(t_i-1,BALL_X:BALL_Y);
    //  x(t_i,BALL_IN_CUP) = 1; % Once in the cup, always in the cup
    //  
    //else
    //  
    //  if x(t_i,HOLD_BALL)
    //    % If the ball is in your hand, it moves along with your hand
    //    x(t_i,BALL_X:BALL_Y) = x(t_i,REF_X:REF_Y);
    //    x(t_i,BALL_XD) = diff(x([t_i-1 t_i],BALL_X))/dt;
    //    x(t_i,BALL_YD) = diff(x([t_i-1 t_i],BALL_Y))/dt;
    //    
    //  else
    //    % If the ball is not in your hand, it simply falls
    //    x(t_i,BALL_XDD) = 0;
    //    x(t_i,BALL_YDD) = -g;
    //    
    //    % Euler integration
    //    x(t_i,BALL_XD:BALL_YD) = x(t_i-1,BALL_XD:BALL_YD) + dt*x(t_i,BALL_XDD:BALL_YDD);
    //    x(t_i,BALL_X:BALL_Y) = x(t_i-1,BALL_X:BALL_Y) + dt*x(t_i,BALL_XD:BALL_YD);
    //    
    //  end
  }
  trajectory.set_misc(y_ball);

  bool overwrite = true;    
  trajectory.saveToFile(traj_filename, overwrite);
  
  delete dmp;
  
  return 0;
}
