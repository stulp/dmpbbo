/**
 * \file demoDmpContextual.cpp
 * \author Freek Stulp
 * \brief  Demonstrates how to initialize, train and integrate a Contextual Dmp.
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

#include "dmp/DmpContextual.hpp"
#include "dmp/DmpContextualOneStep.hpp"
#include "dmp/DmpContextualTwoStep.hpp"
#include "dmp/Trajectory.hpp"

#include "dynamicalsystems/DynamicalSystem.hpp"
#include "dynamicalsystems/ExponentialSystem.hpp"
#include "dynamicalsystems/SigmoidSystem.hpp"
#include "dynamicalsystems/TimeSystem.hpp"
#include "dynamicalsystems/SpringDamperSystem.hpp"

#include "functionapproximators/FunctionApproximator.hpp"
#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/ModelParametersLWR.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"

#include <boost/filesystem.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

/** Get a vector of demonstration trajectories.
 * \param[in] task_parameters_demos Task parameters of the demonstrations
 * \param[in] trajectories A set of demonstration trajectories
 * \param[in] directory_trajectories Directory from which to read trajectories (optional)
 * \return true if reading files was successful, false otherwise
 */
bool getDemoTrajectories(VectorXd& task_parameters_demos, vector<Trajectory>& trajectories, string directory_trajectories="");

/** Main function
 * \param[in] n_args Number of arguments
 * \param[in] args Arguments themselves
 * \return Success of exection. 0 if successful.
 */
int main(int n_args, char** args)
{
  string directory, directory_trajectories;
  bool use_two_step = false;
  if (n_args>1)
    directory = string(args[1]);
  if (n_args>2)
    // If second argument is "2", then use 2-step ContextualDmp instead of 1-step.
    use_two_step = (string(args[2]).compare("2")==0);
  if (n_args>3)
    directory_trajectories = string(args[3]);
  
  cout << "Arguments:" << directory << endl;
  cout << "  directory=" << directory << endl;
  cout << "  use_two_step=" << use_two_step << endl;
  if (!directory_trajectories.empty())
    cout << "  directory_trajectories=" << directory_trajectories << endl;
  
  
  vector<Trajectory> trajectories;
  VectorXd task_parameters_demos;
  
  // getDemoTrajectories(...) is implemented below main(...)
  getDemoTrajectories(task_parameters_demos, trajectories,directory_trajectories);
  
  int n_demos = trajectories.size();
  int n_dims = trajectories[0].dim();
  // Get the number of task parameters
  int n_task_parameters = trajectories[0].dim_misc();
  // Determine time steps for reproduced movement    
  // VectorXd ts = VectorXd::LinSpaced(n_time_steps,0.0,0.5);
  // ts_repro = VectorXd::LinSpaced(n_time_steps_repro,0.0,0.7);
  VectorXd ts_repro = trajectories[0].ts();

  // Make some LWR function approximators
  int n_input_dim = 1;
  if (!use_two_step)
    // The input dimensionality for a 2-step contextual Dmp is 1, i.e. only phase
    // For a 1-step contextual dmp it is 1+n_task_parameters.
    n_input_dim += n_task_parameters;
    
  VectorXi n_bfs_per_dim = VectorXi::Constant(n_input_dim,3);
  n_bfs_per_dim[0] = 50; // Need some more along time dimension;
  
  MetaParametersLWR* meta_parameters = new MetaParametersLWR(n_input_dim,n_bfs_per_dim);
  cout << "MetaParameters of the function approximator:" << endl;
  cout << "   " << *meta_parameters << endl;
  vector<FunctionApproximator*> function_approximators(n_dims);    
  for (int dd=0; dd<n_dims; dd++)
    function_approximators[dd] = new FunctionApproximatorLWR(meta_parameters);

  
  // MAKE A CONTEXTUAL DMP
  DmpContextual* dmp;
  if (use_two_step)
  {
    int n_bfs_per_dim = 4;
    MetaParametersLWR* meta_parameters_ppf = new MetaParametersLWR(n_task_parameters,n_bfs_per_dim);
    FunctionApproximatorLWR* ppf = new FunctionApproximatorLWR(meta_parameters_ppf);
    dmp = new DmpContextualTwoStep(n_dims, function_approximators, ppf);
  }
  else
  {
    dmp = new DmpContextualOneStep(n_dims, function_approximators);
  }
  int n_bfs_per_dim_goal = 4;
  MetaParametersLWR* meta_parameters_ppfg = new MetaParametersLWR(n_task_parameters,n_bfs_per_dim_goal);
  FunctionApproximatorLWR* ppfg = new FunctionApproximatorLWR(meta_parameters_ppfg);
  dmp->set_policy_parameter_function_goal(ppfg);
  
  // TRAIN A CONTEXTUAL DMP
  cout << "Train the Contextual Dmp" << endl;
  bool overwrite = true;
  dmp->train(trajectories, directory, overwrite);
  
  // INTEGRATE THE CONTEXTUAL DMP FOR DIFFERENT TASK PARAMETERS
  int n_repros = 2*n_demos-1;
  VectorXd task_parameters_repros = VectorXd::LinSpaced(n_repros,task_parameters_demos.minCoeff(),task_parameters_demos.maxCoeff());   vector<Trajectory> trajs_reproduced(n_repros);
  
  // Integrate the DMP analytically 
  vector<MatrixXd> forcing_terms(n_repros);
  cout << "Integrate the Contextual Dmp: ";
  for (int i_repro=0; i_repro<n_repros; i_repro++)
  {
    cout << (i_repro+1) << "/" << n_repros << "  ";
    VectorXd cur_task_parameters = task_parameters_repros[i_repro]*MatrixXd::Ones(1,n_task_parameters);
    dmp->set_task_parameters(cur_task_parameters);
    dmp->analyticalSolution(ts_repro,trajs_reproduced[i_repro],forcing_terms[i_repro]);
  }
  cout << endl;
  
  delete dmp;

  
  if (!directory.empty())
  {
    bool overwrite = true;
    
    saveMatrix(directory,"task_parameters_demos.txt",task_parameters_demos, overwrite);
    saveMatrix(directory,"task_parameters_repros.txt",task_parameters_repros, overwrite);

    cout << "Saving demonstrated trajectories to " << directory << endl;
    for (int i_demo=0; i_demo<n_demos; i_demo++)
    {
      stringstream stream;
      stream << "demonstration" << setw(2) << setfill('0') << i_demo << ".txt";
      trajectories[i_demo].saveToFile(directory,stream.str(),overwrite);
    }
    
    cout << "Saving reproduced trajectories to " << directory << endl;
    for (int i_repro=0; i_repro<n_repros; i_repro++)
    {
      stringstream stream;
      stream << "reproduced" << setw(2) << setfill('0') << i_repro << ".txt";
      trajs_reproduced[i_repro].saveToFile(directory,stream.str(),overwrite);
    }
    
    cout << "Saving forcing terms of reproduced trajectories to " << directory << endl;
    for (int i_repro=0; i_repro<n_repros; i_repro++)
    {
      stringstream stream;
      stream << "reproduced_forcingterm" << setw(2) << setfill('0') << i_repro << ".txt";
      saveMatrix(directory,stream.str(),forcing_terms[i_repro],overwrite);
    }
  }

  return 0;
}


bool getDemoTrajectories(VectorXd& task_parameters_demos, vector<Trajectory>& trajectories,string directory_trajectories)
{

  int n_dims;
  int n_demos;
  if (directory_trajectories.empty())
  {
    int n_time_steps = 51;
    int n_time_steps_repro = 71;
    n_dims = 1;
    n_demos = 5;
    
    // Make some trajectories
    VectorXd ts = VectorXd::LinSpaced(n_time_steps,0.0,0.5);
    VectorXd ts_repro = VectorXd::LinSpaced(n_time_steps_repro,0.0,0.7);
    
  
    VectorXd y_first = VectorXd::Zero(n_dims);
    VectorXd y_last  = VectorXd::Ones(n_dims);
    double viapoint_time = 0.25;
  
    task_parameters_demos = VectorXd::LinSpaced(n_demos,0.2,0.8);
    
    trajectories.resize(n_demos);
    for (int i_demo=0; i_demo<n_demos; i_demo++)
    {
      VectorXd task_parameters = task_parameters_demos[i_demo]*MatrixXd::Ones(1,n_dims);
      
      VectorXd y_yd_ydd_viapoint = VectorXd::Zero(3*n_dims);
      y_yd_ydd_viapoint.segment(0*n_dims,n_dims) = task_parameters.row(0); // y         y_yd_ydd_viapoint.segment(1*n_dims,n_dims) = VectorXd::Constant(n_dims,1.0); // yd    y_yd_ydd_viapoint.segment(2*n_dims,n_dims) = VectorXd::Constant(n_dims,0.0); // ydd

      y_last[0] = 1.5-task_parameters[0];
      
      trajectories[i_demo] = Trajectory::generatePolynomialTrajectoryThroughViapoint(ts,y_first,y_yd_ydd_viapoint,viapoint_time,y_last); 
      
      trajectories[i_demo].set_misc(task_parameters_demos.segment(i_demo,1));
    }
    
  }
  else
  {
    // Read some trajectories from a directory
    int n_dims_misc = 1;
    n_demos=0;
    bool go_on=true;
    while (go_on)
    {
      stringstream stream;
      stream << directory_trajectories << "/" << "traj_and_task_params_" << setfill('0') << setw(3) << (n_demos+1) << ".txt";
      string filename = stream.str();    
      
      if (boost::filesystem::exists(filename))
      {
        Trajectory cur_traj = Trajectory::readFromFile(stream.str(), n_dims_misc);
        //cout << cur_traj << endl;
        n_dims = cur_traj.dim();
        trajectories.push_back(cur_traj);
        n_demos++;
      }
      else 
      {
        go_on = false;
      }
    }
    
    
    // Gather the task parameters
    task_parameters_demos = VectorXd(n_demos);
    for (int i_demo=0; i_demo<n_demos; i_demo++)
    {
      task_parameters_demos[i_demo] = trajectories[i_demo].misc()(0,0);
    }

  }
  return true;
}
