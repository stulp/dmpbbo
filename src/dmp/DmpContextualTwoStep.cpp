/**
 * @file DmpContextualTwoStep.cpp
 * @brief  Contextual Dmp class source file.
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

#include "dmp/Dmp.hpp"
#include "dmp/DmpContextualTwoStep.hpp"

#include "dmp/Trajectory.hpp"
#include "functionapproximators/FunctionApproximator.hpp"
#include "functionapproximators/FunctionApproximatorLWR.hpp"

#include <boost/serialization/vector.hpp>

#include <iostream>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

namespace DmpBbo {

DmpContextualTwoStep::DmpContextualTwoStep(int n_dims_dmp, std::vector<FunctionApproximator*> function_approximators, FunctionApproximator* policy_parameter_function, DmpType dmp_type) 
:  DmpContextual(n_dims_dmp, function_approximators, dmp_type)
{
  policy_parameter_function_ = vector<vector<FunctionApproximator*> >(dim_orig());
  for (int dd=0; dd<dim_orig(); dd++)
  { 
    policy_parameter_function_[dd] = vector<FunctionApproximator*>(1);
    policy_parameter_function_[dd][0] = policy_parameter_function->clone();
  }
}

// Overloads in DMP computeFunctionApproximatorOutput
void DmpContextualTwoStep::computeFunctionApproximatorOutput(const MatrixXd& phase_state, MatrixXd& fa_output) const
{
  int n_time_steps = phase_state.rows(); 
  fa_output.resize(n_time_steps,dim_orig());
  fa_output.fill(0.0);
  
  if (task_parameters_.rows()==0)
  {
    // When the task parameters are not set, we cannot compute the output of the function approximator.
    return;
  }
  
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
  
  //int n_task_parameters = task_parameters.cols();
  
  VectorXd model_parameters;
  MatrixXd output(1,1);
  for (int dd=0; dd<dim_orig(); dd++)
  { 
    int n_parameters = function_approximator(dd)->getParameterVectorSelectedSize();
    model_parameters.resize(n_parameters);
    for (int pp=0; pp<n_parameters; pp++)
    {
      policy_parameter_function_[dd][pp]->predict(task_parameters,output);
      model_parameters[pp] = output(0,0);
    }
    function_approximator(dd)->setParameterVectorSelected(model_parameters);
  }

  // The parameters of the function_approximators have been set, get their outputs now.  
  for (int dd=0; dd<dim_orig(); dd++)
  {
    function_approximator(dd)->predict(phase_state,output);
    if (output.size()>0)
    {
      fa_output.col(dd) = output;
    }
  }

}

bool DmpContextualTwoStep::isTrained(void) const
{
  for (int dd=0; dd<dim_orig(); dd++)
    for (unsigned int pp=0; pp<policy_parameter_function_[dd].size(); pp++)
      if (!policy_parameter_function_[dd][pp]->isTrained())
        return false;
      
  return true;   
}

void  DmpContextualTwoStep::train(const vector<Trajectory>& trajectories, const vector<MatrixXd>& task_parameters, string save_directory, bool overwrite)
{
  // Here's the basic structure of this function
  // 1) Do some checks
  // 2) Train a separate Dmp for each demonstration, and get the resulting model parameters
  // 3) Gather all task parameter values for all demonstrations
  // 4) Train the policy parameter function for each dimension and each model parameter
  
  
  //-----------------------------------------------------
  // 1) Do some checks
  
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

  //-----------------------------------------------------
  // 2) Train a separate Dmp for each demonstration, and get the resulting model parameters
  std::set<std::string> selected;
  selected.insert("offsets");
  selected.insert("slopes");
  
  MatrixXd cur_task_parameters;
  VectorXd cur_model_parameters;// todo Remove redundant tmp variable
  vector<MatrixXd> all_model_parameters(n_demonstrations); 
  for (unsigned int i_demo=0; i_demo<n_demonstrations; i_demo++)
  {
    
    string save_directory_demo;
    if (!save_directory.empty())
      save_directory_demo = save_directory + "/demo" + to_string(i_demo);
    
    Dmp::train(trajectories[i_demo],save_directory_demo,overwrite);
    
    for (int i_dim=0; i_dim<dim_orig(); i_dim++)
    {

      // todo Should be argument of constructor
      function_approximator(i_dim)->setSelectedParameters(selected); 
  
      function_approximator(i_dim)->getParameterVectorSelected(cur_model_parameters);
      //cout << cur_model_parameters << endl;
      if (i_demo==0)
        all_model_parameters[i_dim].resize(n_demonstrations,cur_model_parameters.size());
      else
        assert(cur_model_parameters.size()==all_model_parameters[i_dim].cols());

      all_model_parameters[i_dim].row(i_demo) = cur_model_parameters;
      
    }
  }

   
  //-----------------------------------------------------
  // 3) Gather all task parameter values for all demonstrations
  
  // Gather task parameters in a matrix
  int n_task_parameters = task_parameters[0].cols();
  // This is the first time task_parameters_ is set, because this is the first time we know 
  // n_task_parameters.
  // We set it so that set_task_parameters can check if task_parameters_.cols()==n_task_parameters
  task_parameters_ = MatrixXd::Zero(1,n_task_parameters);
  VectorXd cur_task_parameters_t0;
  
  MatrixXd inputs(n_demonstrations,n_task_parameters);
  for (unsigned int i_demo=0; i_demo<n_demonstrations; i_demo++)
  {
    // These are the task parameters for the current demonstration at t=0
    cur_task_parameters_t0 = task_parameters[i_demo].row(0);
    
    // Task parameter may not change over time for 2-Step contextual DMP
    // Start comparison to i_time=0 at i_time=1
    for (int i_time=1; i_time<task_parameters[i_demo].rows(); i_time++)
    {
      if ( (cur_task_parameters_t0.array() != task_parameters[i_demo].row(i_time).array()).any())
      {
        cerr << __FILE__ << ":" << __LINE__ << ":";
        cerr << "WARNING. For DmpContextualTwoStep, task parameters may not vary over time during training. Using task parameters at t=0 only." << endl;
      }
    }

    // Take the first row, i.e. at time_i = 0. We checked above if they are constant over time.
    inputs.row(i_demo) = cur_task_parameters_t0;
  }

  //-----------------------------------------------------
  // 4) Train the policy parameter function for each dimension and each model parameter
  
  // Input to policy parameter functions: task_parameters
  // Target for each policy parameter function: all_model_parameters.col(param)
  
  for (int i_dim=0; i_dim<dim_orig(); i_dim++)
  {
    int n_pol_pars = all_model_parameters[i_dim].cols();
    for (int i_pol_par=1; i_pol_par<n_pol_pars; i_pol_par++)
    {
      policy_parameter_function_[i_dim].push_back(policy_parameter_function_[i_dim][0]->clone());
      //cout << *(policy_parameter_function_[i_dim][i_pol_par]) << endl;
    }

    for (int i_pol_par=0; i_pol_par<n_pol_pars; i_pol_par++)
    {
      MatrixXd targets = all_model_parameters[i_dim].col(i_pol_par);
      //cout << "_________________" << endl;
      //cout << inputs.transpose() << endl << endl;
      //cout << targets.transpose() << endl;
      
      string save_directory_cur;
      if (!save_directory.empty())
          save_directory_cur = save_directory + "/dim" + to_string(i_dim) + "_polpar" + to_string(i_pol_par);
      
      policy_parameter_function_[i_dim][i_pol_par]->train(inputs,targets,save_directory_cur,overwrite);
    
    }
  }
  
  
}

template<class Archive>
void DmpContextualTwoStep::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(DmpContextual);
  
  ar & BOOST_SERIALIZATION_NVP(policy_parameter_function_);

}

}
