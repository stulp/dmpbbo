/**
 * \file demoOptimizationDmpParallel.cpp
 * \author Freek Stulp
 * \brief  Demonstrates how to run multiple evolution strategies in parallel to optimize a Dmp.
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

#include "dmp_bbo/runOptimizationTask.hpp"
#include "dmp_bbo/tasks/TaskViapoint.hpp"
#include "dmp_bbo/TaskSolverDmp.hpp"

#include "dmp/Dmp.hpp"
#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/ModelParametersLWR.hpp"

#include "bbo/DistributionGaussian.hpp"
#include "bbo/Updater.hpp"
#include "bbo/updaters/UpdaterCovarDecay.hpp"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

/** Main function
 * \param[in] n_args Number of arguments
 * \param[in] args Arguments themselves
 * \return Success of exection. 0 if successful.
 */
int main(int n_args, char* args[])
{
  // If program has an argument, it is a directory to which to save files too (or --help)
  string directory;
  if (n_args>1)
  {
    if (string(args[1]).compare("--help")==0)
    {
      cout << "Usage: " << args[0] << " [directory]         (directory: optional directory to save data to)" << endl;
      return 0;
    }
    else
    {
      directory = string(args[1]);
    }
  }

  // Make the task
  int n_dim=2;
  
  VectorXd viapoint = VectorXd::LinSpaced(n_dim,1.5,2);
  double viapoint_time = 0.2;
  TaskViapoint* task = new TaskViapoint(viapoint,viapoint_time);
  
  // Some DMP parameters
  double tau = 0.6;
  VectorXd y_init = VectorXd::Constant(n_dim,1.0);
  VectorXd y_attr = VectorXd::Constant(n_dim,3.0);
 
  int n_basis_functions = 4;
  VectorXd centers = VectorXd::LinSpaced(n_basis_functions,0,1);
  VectorXd widths  = VectorXd::Constant(n_basis_functions,0.2);
  VectorXd slopes  = VectorXd::Zero(n_basis_functions);
  VectorXd offsets = VectorXd::Zero(n_basis_functions);
  ModelParametersLWR* model_parameters = new ModelParametersLWR(centers,widths,slopes,offsets);
  vector<FunctionApproximator*> function_approximators(n_dim);
  for (int i_dim=0; i_dim<n_dim; i_dim++)
    function_approximators[i_dim] = new FunctionApproximatorLWR(model_parameters);
  
  Dmp* dmp = new Dmp(tau, y_init, y_attr, function_approximators, Dmp::KULVICIUS_2012_JOINING);

  // Make the task solver
  double integrate_dmp_beyond_tau_factor = 1.25;
  double dt=0.01;
  set<string> parameters_to_optimize;
  parameters_to_optimize.insert("offsets");
  
  TaskSolver* task_solver = new TaskSolverDmp(dmp, parameters_to_optimize, dt, integrate_dmp_beyond_tau_factor);

  // Make the initial distribution
  vector<VectorXd> mean_init_vec;
  dmp->getParameterVectorSelected(mean_init_vec);
  
  vector<DistributionGaussian*> distributions(n_dim);
  for (int i_dim=0; i_dim<n_dim; i_dim++)
  {
    //cout << mean_init_vec[i_dim].transpose() << endl;
    VectorXd mean_init = mean_init_vec[i_dim];
  
    MatrixXd covar_init = 1000.0*MatrixXd::Identity(mean_init.size(),mean_init.size());
    
    distributions[i_dim] = new DistributionGaussian(mean_init,covar_init);
  }


  // Make the parameter updater
  double eliteness = 10;
  double covar_decay_factor = 0.9;
  string weighting_method("PI-BB");
  Updater* updater = new UpdaterCovarDecay(eliteness, covar_decay_factor, weighting_method);
  
  // Run the optimization
  int n_updates = 40;
  int n_samples_per_update = 15;
  bool overwrite = true;
  runOptimizationParallelDeprecated(task, task_solver, distributions, updater, n_updates, n_samples_per_update,directory,overwrite);
  
}