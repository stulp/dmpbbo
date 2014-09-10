/**
 * @file DmpContextualOneStep.cpp
 * @brief  Contextual Dmp class header file.
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

#include "dmp/DmpContextualOneStep.hpp"

#include <iomanip>
#include <iostream>
#include <eigen3/Eigen/Core>

#include "dmp/Trajectory.hpp"
#include "functionapproximators/FunctionApproximator.hpp"

using namespace std;
using namespace Eigen;

namespace DmpBbo {

DmpContextualOneStep::DmpContextualOneStep(int n_dims_dmp, std::vector<FunctionApproximator*> function_approximators,
  DmpType dmp_type) 
:  DmpContextual(n_dims_dmp, function_approximators, dmp_type)
{
}
  
// Overloads in DMP computeFunctionApproximatorOutput
void DmpContextualOneStep::computeFunctionApproximatorOutput(const MatrixXd& phase_state, MatrixXd& fa_output) const
{
  int n_time_steps = phase_state.rows(); 
  fa_output.resize(n_time_steps,dim_orig());
  fa_output.fill(0.0);
  
  MatrixXd task_parameters = task_parameters_;
  if (task_parameters.rows()==1)
  { 
    task_parameters = task_parameters.row(0).replicate(n_time_steps,1).eval();
  }
  else if (task_parameters.cols()==1)
  {
    task_parameters = task_parameters.col(0).transpose().replicate(n_time_steps,1).eval();
  }


  
  assert(n_time_steps==task_parameters.rows());
  
  int n_task_parameters = task_parameters.cols();
  MatrixXd fa_input(n_time_steps,n_task_parameters+1);
  fa_input << phase_state, task_parameters;
  
  
  MatrixXd output(n_time_steps,1);
  for (int dd=0; dd<dim_orig(); dd++)
  {
    if (function_approximator(dd)!=NULL)
    {
      if (function_approximator(dd)->isTrained()) 
      {
        function_approximator(dd)->predict(fa_input,output);
        if (output.size()>0)
        {
          fa_output.col(dd) = output;
        }
      }
    }
  }
}

void  DmpContextualOneStep::train(const vector<Trajectory>& trajectories, const vector<MatrixXd>& task_parameters, string save_directory, bool overwrite)
{
  // Check if inputs are of the right size.
  unsigned int n_demonstrations = trajectories.size();
  assert(n_demonstrations==task_parameters.size());
  
  // Then check if the trajectories have the same duration and initial/final state
  // Later on, if they are not the same, they should be learned also.
  checkTrainTrajectories(trajectories);

  // Set tau, initial_state and attractor_state from the trajectories 
  set_tau(trajectories[0].duration());
  set_initial_state(trajectories[0].initial_y());
  set_attractor_state(trajectories[0].final_y());
  
  MatrixXd all_fa_inputs(0,0);
  MatrixXd all_fa_targets(0,0);  
  int n_task_parameters = -1;
  int n_time_steps_total = 0;

  VectorXd cur_fa_input_phase;
  MatrixXd cur_fa_target;
  MatrixXd cur_task_parameters;
  for (unsigned int i_demo=0; i_demo<n_demonstrations; i_demo++)
  {
    cur_task_parameters = task_parameters[i_demo]; 
    if (i_demo==0)
    {
      n_task_parameters = cur_task_parameters.cols();
      
      // This is the first time task_parameters_ is set, because this is the first time we know 
      // n_task_parameters.
      // We set it so that set_task_parameters can check if task_parameters_.cols()==n_task_parameters
      task_parameters_ = MatrixXd::Zero(1,n_task_parameters);
    }
    else
    {
      assert(n_task_parameters==cur_task_parameters.cols());
    }
    
    int n_time_steps = trajectories[i_demo].length();
    n_time_steps_total += n_time_steps;

    // Make sure cur_task_parameters has n_time_steps. Copy if it doesn't.
    if (cur_task_parameters.rows()==1 && n_time_steps>1) {      
      MatrixXd cur_task_parameters_tmp = cur_task_parameters.replicate(n_time_steps,1);
      cur_task_parameters = cur_task_parameters_tmp;
    }
    assert(n_time_steps==cur_task_parameters.rows());
    
    computeFunctionApproximatorInputsAndTargets(trajectories[i_demo], cur_fa_input_phase, cur_fa_target);
    
    all_fa_inputs.conservativeResize(n_time_steps_total,1+n_task_parameters);
    
    all_fa_inputs.bottomLeftCorner(n_time_steps,1) = cur_fa_input_phase;
    all_fa_inputs.bottomRightCorner(n_time_steps,n_task_parameters) = cur_task_parameters;

    all_fa_targets.conservativeResize(n_time_steps_total,dim_orig());
    all_fa_targets.bottomRows(n_time_steps) = cur_fa_target;
    
  }

  // We have all inputs and targets. Now train the function approximator for each dimension.
  
  for (int dd=0; dd<dim_orig(); dd++)
  {
    // This is just boring stuff to figure out if and where to store the results of training
    string save_directory_dim;
    if (!save_directory.empty())
    {
      if (dim_orig()==1)
        save_directory_dim = save_directory;
      else
        save_directory_dim = save_directory + "/dim" + to_string(dd);
    }
    
    VectorXd fa_target = all_fa_targets.col(dd);
    function_approximator(dd)->train(all_fa_inputs,fa_target,save_directory_dim,overwrite);
  }
  
}

}
