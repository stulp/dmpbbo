/**
 * @file DmpContextual.cpp
 * @brief  DmpContextual class source file.
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
#define EIGEN2_SUPPORT
#include "dmp/DmpContextual.hpp"

#include "dmp/Trajectory.hpp"

#include <iostream>
#include <eigen3/Eigen/Core>


using namespace std;
using namespace Eigen;

namespace DmpBbo {

DmpContextual::DmpContextual(int n_dims_dmp, vector<FunctionApproximator*> function_approximators, DmpType dmp_type) 
:  Dmp(n_dims_dmp, function_approximators, dmp_type)
{
   
}

// In the end, all of the below train(...) variants call the pure virtual function
// virtual void  train(const std::vector<Trajectory>& trajectories, const std::vector<Eigen::MatrixXd>& task_parameters, std::string save_directory, bool overwrite) = 0;


void  DmpContextual::train(const vector<Trajectory>& trajectories, const vector<MatrixXd>& task_parameters, string save_directory)
{
  bool overwrite = false;
  train(trajectories, task_parameters, save_directory, overwrite);
}

void  DmpContextual::train(const vector<Trajectory>& trajectories, const vector<MatrixXd>& task_parameters)
{
  bool overwrite=false;
  string save_directory("");
  train(trajectories, task_parameters, save_directory, overwrite);
}

void  DmpContextual::train(const vector<Trajectory>& trajectories, string save_directory, bool overwrite)
{
  vector<MatrixXd> task_parameters(trajectories.size());
  for (unsigned int i_traj=0; i_traj<trajectories.size(); i_traj++)
  {
    task_parameters[i_traj] = trajectories[i_traj].misc();
    assert(task_parameters[i_traj].cols()>0);
    if (i_traj>0)
      assert(task_parameters[i_traj].cols() ==  task_parameters[i_traj].cols());
  }
  train(trajectories, task_parameters, save_directory, overwrite);
}

void  DmpContextual::train(const vector<Trajectory>& trajectories, string save_directory)
{
  bool overwrite=false;
  train(trajectories, save_directory, overwrite);
}

void  DmpContextual::train(const vector<Trajectory>& trajectories)
{
  bool overwrite=false;
  string save_directory("");
  train(trajectories, save_directory, overwrite);
}


void  DmpContextual::checkTrainTrajectories(const vector<Trajectory>& trajectories)
{
  // Check if inputs are of the right size.
  unsigned int n_demonstrations = trajectories.size();
  
  // Then check if the trajectories have the same duration and initial/final state
  double first_duration = trajectories[0].duration();
  VectorXd first_y_init = trajectories[0].initial_y();
  VectorXd first_y_attr = trajectories[0].final_y();  
  for (unsigned int i_demo=1; i_demo<n_demonstrations; i_demo++)
  {
    // Difference in tau
    if (fabs(first_duration-trajectories[i_demo].duration())>10e-4)
    {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "WARNING: Duration of demonstrations differ (" << first_duration << "!=" << trajectories[i_demo].duration() << ")" << endl;
    }
    
    // Difference between initial states
    double sum_abs_diff = (first_y_init.array()-trajectories[i_demo].initial_y().array()).abs().sum();
    if (sum_abs_diff>10e-7)
    {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "WARNING: Final states of demonstrations differ ( [" << first_y_init.transpose() << "] != [ " << trajectories[i_demo].initial_y().transpose() << "] )" << endl;
    }
    
    // Difference between final states
    sum_abs_diff = (first_y_attr.array()-trajectories[i_demo].final_y().array()).abs().sum();
    if (sum_abs_diff>10e-7)
    {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "WARNING: Final states of demonstrations differ ( [" << first_y_attr.transpose() << "] != [ " << trajectories[i_demo].final_y().transpose() << "] )" << endl;
    }
    
  }
}

}
