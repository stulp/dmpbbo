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

#include "dmp/DmpContextual.hpp"

#include "dmp/Trajectory.hpp"
#include "functionapproximators/FunctionApproximator.hpp"

#include <assert.h>
#include <iostream>
#include <eigen3/Eigen/Core>


using namespace std;
using namespace Eigen;

namespace DmpBbo {

DmpContextual::DmpContextual(int n_dims_dmp, std::vector<FunctionApproximator*> function_approximators, DmpType dmp_type) 
:  Dmp(n_dims_dmp, function_approximators, dmp_type)
{
  policy_parameter_function_goal_ = std::vector<FunctionApproximator*>(0);
  policy_parameter_function_duration_ = NULL;
}

void DmpContextual::set_policy_parameter_function_goal(FunctionApproximator* function_approximator)
{
  // Make clones, one for each of the dimensions of the goal
  policy_parameter_function_goal_ = vector<FunctionApproximator*>(dim_orig());
  for (int dd=0; dd<dim_orig(); dd++)
  { 
    policy_parameter_function_goal_[dd] = function_approximator->clone();
  }
}

void DmpContextual::set_policy_parameter_function_duration(FunctionApproximator* function_approximator)
{
  // Output of this function approximator should always be 1D (duration is time is 1D)
  assert(function_approximator->getExpectedOutputDim()==1);

  policy_parameter_function_duration_ = function_approximator->clone();  
}

// In the end, all of the below train(...) variants call the pure virtual function
// virtual void  train(const std::vector<Trajectory>& trajectories, const std::vector<Eigen::MatrixXd>& task_parameters, std::string save_directory, bool overwrite) = 0;


void  DmpContextual::train(const std::vector<Trajectory>& trajectories, const std::vector<Eigen::MatrixXd>& task_parameters, std::string save_directory)
{
  bool overwrite = false;
  train(trajectories, task_parameters, save_directory, overwrite);
}

void  DmpContextual::train(const std::vector<Trajectory>& trajectories, const std::vector<Eigen::MatrixXd>& task_parameters)
{
  bool overwrite=false;
  string save_directory("");
  train(trajectories, task_parameters, save_directory, overwrite);
}


void  DmpContextual::trainLocal(const std::vector<Trajectory>& trajectories, const std::vector<Eigen::MatrixXd>& task_parameters, std::string save_directory, bool overwrite)
{
  if (policy_parameter_function_goal_.size()>0 || policy_parameter_function_duration_!=NULL)
  {
    unsigned int n_demonstrations = trajectories.size();
    assert(n_demonstrations==task_parameters.size());  
    
    // Gather task parameters in a matrix
    int n_task_parameters = task_parameters[0].cols();
    MatrixXd inputs(n_demonstrations,n_task_parameters);
    for (unsigned int i_demo=0; i_demo<n_demonstrations; i_demo++)
      // Take the first row, i.e. at time_i = 0. 
      inputs.row(i_demo) = task_parameters[i_demo].row(0);

    if (policy_parameter_function_goal_.size()>0)
    {
      // Gather goals (target values for approximation) in one matrix
      MatrixXd targets(n_demonstrations,dim_orig());
      for (unsigned int i_demo=0; i_demo<n_demonstrations; i_demo++)
        targets.row(i_demo) = trajectories[i_demo].final_y();
      
      //cout << "  inputs=" << inputs << endl;
      //cout << "  targets=" << targets << endl;
      //cout << "  targets.col(0)=" << targets.col(0) << endl;
      
      for (int i_dim=0; i_dim<dim_orig(); i_dim++)
      {
        string save_directory_cur;
        if (!save_directory.empty())
            save_directory_cur = save_directory + "/dim" + to_string(i_dim) + "_goal";
        
        policy_parameter_function_goal_[i_dim]->train(inputs,targets.col(i_dim),save_directory_cur,overwrite);
      }
    }
    
    if (policy_parameter_function_duration_!=NULL)
    {
      // Gather durations (target values for approximation) in one matrix
      VectorXd targets(n_demonstrations);
      for (unsigned int i_demo=0; i_demo<n_demonstrations; i_demo++)
        targets(i_demo) = trajectories[i_demo].duration();
      
      //cout << "  inputs=" << inputs << endl;
      //cout << "  targets=" << targets << endl;
      
      string save_directory_cur;
      if (!save_directory.empty())
          save_directory_cur = save_directory + "/_duration";
      
      policy_parameter_function_duration_->train(inputs,targets,save_directory_cur,overwrite);
    }
    
  }
  
  // Train the rest, i.e. the parameters for the forcing term.
  train(trajectories, task_parameters, save_directory, overwrite);
}

void DmpContextual::set_task_parameters(const MatrixXd& task_parameters)
{
  assert(task_parameters_.cols()==task_parameters.cols());
  task_parameters_ = task_parameters; // This will be used for the forcing term later
  
  // Compute new attractor state now, if necessary.
  if (policy_parameter_function_goal_.size()>0)
  { 
    // TODO: Check size of task_parameters
    VectorXd goal(dim_orig());
    MatrixXd outputs;
    for (int i_dim=0; i_dim<dim_orig(); i_dim++)
    {
      policy_parameter_function_goal_[i_dim]->predict(task_parameters,outputs);
      goal(i_dim) = outputs(0,0);
    }
    set_attractor_state(goal);
  }
  
  // Compute new duration now, if necessary.
  if (policy_parameter_function_duration_!=NULL)
  {
    // TODO: Check size of task_parameters
    MatrixXd duration_as_matrix;
    policy_parameter_function_duration_->predict(task_parameters,duration_as_matrix);
    set_tau(duration_as_matrix(0,0));
  }
  
}

void  DmpContextual::train(const std::vector<Trajectory>& trajectories, std::string save_directory, bool overwrite)
{
  vector<MatrixXd> task_parameters(trajectories.size());
  for (unsigned int i_traj=0; i_traj<trajectories.size(); i_traj++)
  {
    task_parameters[i_traj] = trajectories[i_traj].misc();
    assert(task_parameters[i_traj].cols()>0);
    if (i_traj>0)
      assert(task_parameters[i_traj].cols() ==  task_parameters[i_traj].cols());
  }
  trainLocal(trajectories, task_parameters, save_directory, overwrite);
}

void  DmpContextual::train(const std::vector<Trajectory>& trajectories, std::string save_directory)
{
  bool overwrite=false;
  train(trajectories, save_directory, overwrite);
}

void  DmpContextual::train(const std::vector<Trajectory>& trajectories)
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
        if (policy_parameter_function_duration_==NULL) 
        {
          cerr << __FILE__ << ":" << __LINE__ << ":";
          cerr << "WARNING: Duration of demonstrations differ (" << first_duration << "!=" << trajectories[i_demo].duration() << ")" << endl;
          cerr << "         See DmpContextual::set_policy_parameter_function_duration(...) on how to fix this. " << endl;
        }
      }
    
    // Difference between initial states
    double sum_abs_diff = (first_y_init.array()-trajectories[i_demo].initial_y().array()).abs().sum();
    if (sum_abs_diff>10e-7)
    {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "WARNING: Initial states of demonstrations differ ( [" << first_y_init.transpose() << "] != [ " << trajectories[i_demo].initial_y().transpose() << "] )" << endl;
    }
    
    // Difference between final states
    sum_abs_diff = (first_y_attr.array()-trajectories[i_demo].final_y().array()).abs().sum();
    if (sum_abs_diff>10e-7 && policy_parameter_function_goal_.size()==0)
    {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "WARNING: Final states of demonstrations differ ( [" << first_y_attr.transpose() << "] != [ " << trajectories[i_demo].final_y().transpose() << "] )" << endl;
      cerr << "         See DmpContextual::set_policy_parameter_function_goal(...) on how to fix this. " << endl;
    }
    
  }
}

}
