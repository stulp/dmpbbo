/**
 * \file demoOptimizationDmpArm2D.cpp
 * \author Freek Stulp
 * \brief  Demonstrates how to run an evolution strategy to optimize a Dmp, on a task with a viapoint task with a N-DOF arm in a 2D space.
 *
 * \ingroup Demos
 * \ingroup DMP_BBO                
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

#include <string>
#include <set>
#include <eigen3/Eigen/Core>

#include "dmp_bbo/tasks/TaskViapointArm2D.hpp"
#include "dmp_bbo/tasks/TaskSolverDmpArm2D.hpp"
#include "dmp_bbo/runOptimizationTask.hpp"

#include "dmp/Dmp.hpp"
#include "functionapproximators/MetaParametersRBFN.hpp"
#include "functionapproximators/FunctionApproximatorRBFN.hpp"

#include "bbo/DistributionGaussian.hpp"
#include "bbo/Updater.hpp"
#include "bbo/updaters/UpdaterCovarDecay.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

/** Main function
 * \param[in] n_args Number of arguments
 * \param[in] args Arguments themselves
 * \return Success of exection. 0 if successful.
 * \todo To focus on the demo code, it would be nice to have the processing of arguments in a separate function.
 */
int main(int n_args, char* args[])
{
  int n_dofs = 1;
  string directory;
  if (n_args!=3)
  {
      cout << "Usage: " << args[0] << " n_dofs directory" << endl;
      return -1;
  }
  else
  {
    n_dofs = atoi(args[1]);
    directory = string(args[2]);
  }

  if (!boost::filesystem::exists(directory))
  {
    cerr << "Directory '" << directory << "' does not exist." << endl;
    cerr << "HINT: The preferred way to run this demo is by calling ";
    cerr << "python demos/dmp_bbo/" << (args[0]+2) <<"Wrapper.py, "; // +2 removes leading "./" 
    cerr << "rather than this binary." << endl; 
    cerr << "Abort." << endl;
    return -1;
  }

  // Generate a training trajectory for the Dmp
  VectorXd initial_angles;
  TaskSolverDmpArm2D::getInitialAngles(n_dofs,initial_angles);
  VectorXd final_angles;
  TaskSolverDmpArm2D::getFinalAngles(n_dofs,final_angles);
  double tau = 1.0;
  double dt=0.01;
  int n_time_steps = 1 + (int)(tau/dt);
  VectorXd ts = VectorXd::LinSpaced(n_time_steps,0.0,tau);
  Trajectory trajectory = Trajectory::generateMinJerkTrajectory(ts, initial_angles, final_angles);
  
  // Make the initial function approximators 
  int n_basis_functions = 8;
  double intersection_height = 0.7;
  MetaParametersRBFN* meta_parameters = new MetaParametersRBFN(1, n_basis_functions, intersection_height);
  vector<FunctionApproximator*> function_approximators(n_dofs);
  for (int i_dof=0; i_dof<n_dofs; i_dof++)
    function_approximators[i_dof] = new FunctionApproximatorRBFN(meta_parameters);
  
  // Initialize and train the Dmp
  Dmp* dmp = new Dmp(n_dofs, function_approximators, Dmp::KULVICIUS_2012_JOINING);
  dmp->train(trajectory,directory+"/dmptraining",true);
  
  
  // Make the task solver
  set<string> parameters_to_optimize;
  parameters_to_optimize.insert("weights");
  double integrate_dmp_beyond_tau_factor=1.5;
  bool use_normalized_parameter=false;
  
  VectorXd link_lengths = VectorXd::Constant(n_dofs,1.0/n_dofs);
  
  TaskSolverDmpArm2D* task_solver = new TaskSolverDmpArm2D(dmp,link_lengths,parameters_to_optimize,
                                       dt,integrate_dmp_beyond_tau_factor,use_normalized_parameter);
  // task_solver->set_perturbation(1.0); // Add perturbations
  
  VectorXd viapoint = VectorXd::Constant(2,0.5);
  TaskViapointArm2D* task = new TaskViapointArm2D(n_dofs,viapoint);
  
  // Make the initial distribution
  VectorXd mean_init;
  dmp->getParameterVectorSelected(mean_init);
  
  MatrixXd covar_init = 1000.0*MatrixXd::Identity(mean_init.size(),mean_init.size());

  DistributionGaussian* distribution = new DistributionGaussian(mean_init,covar_init);

  // Make the parameter updater
  double eliteness = 10;
  double covar_decay_factor = 0.8;
  string weighting_method("PI-BB");
  Updater* updater = new UpdaterCovarDecay(eliteness, covar_decay_factor, weighting_method);
  
  
  
  // Run the optimization
  int n_updates = 40;
  int n_samples_per_update = 15;
  bool overwrite = true;
  runOptimizationTask(task, task_solver, distribution, updater, n_updates, n_samples_per_update,directory,overwrite);
  
}