/**
 * \file demoOptimizationTask.cpp
 * \author Freek Stulp
 * \brief  Demonstrates how to run an evolution strategy to optimize the parameters of a quadratic function, implemented as a Task and TaskSolver.
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

#include "dmp_bbo/Task.hpp"
#include "dmp_bbo/TaskSolver.hpp"
#include "dmp_bbo/ExperimentBBO.hpp"
#include "dmp_bbo/runOptimizationTask.hpp"

#include "bbo/DistributionGaussian.hpp"

#include "bbo/updaters/UpdaterMean.hpp"
#include "bbo/updaters/UpdaterCovarDecay.hpp"
#include "bbo/updaters/UpdaterCovarAdaptation.hpp"

#include <iomanip> 
#include <fstream> 
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

/** Target function \f$ y = a*x^2 + c \f$
 *  \param[in] a a in \f$ y = a*x^2 + c \f$
 *  \param[in] c c in \f$ y = a*x^2 + c \f$
 *  \param[in] inputs x in \f$ y = a*x^2 + c \f$
 *  \param[out] outputs y in \f$ y = a*x^2 + c \f$
 */
void targetFunction(double a, double c, const VectorXd& inputs, VectorXd& outputs)
{
  // Compute a*x^2 + c
  outputs =  (a*inputs).array().square() + c;
}

/**
 * The task is to choose the parameters a and c such that the function \f$ y = a*x^2 + c \f$ best matches a set of target values y_target for a set of input values x
 */
class DemoTaskApproximateQuadraticFunction : public Task
{
public:
  /** Constructor
   *  \param[in] a a in \f$ y = a*x^2 + c \f$
   *  \param[in] c c in \f$ y = a*x^2 + c \f$
   *  \param[in] inputs x in \f$ y = a*x^2 + c \f$
   */
  DemoTaskApproximateQuadraticFunction(double a, double c, const VectorXd& inputs,double regularization_weight) 
  {
    inputs_ = inputs;
    targetFunction(a,c,inputs_,targets_);
    regularization_weight_ = regularization_weight;
  }
  
  /** Cost function
   * \param[in] cost_vars y in \f$ y = a*x^2 + c \f$
   * \param[in] sample The sample from which cost_vars was generated. Required for regularization.
   * \param[in] task_parameters Ignored
   * \param[out] cost Cost of the rollout. 
   */
  void evaluateRollout(const MatrixXd& cost_vars, const VectorXd& sample, const VectorXd& task_parameters, VectorXd& cost) const
  {
    VectorXd diff_square = (cost_vars.array()-targets_.array()).square();
    cost.resize(3);
    cost[1] = diff_square.mean();
    cost[2] = regularization_weight_*sqrt(sample.array().pow(2).sum());
    cost[0] = cost[1] + cost[2];
  }
  
  unsigned int getNumberOfCostComponents(void) const
  { 
    return 2;
  };
  
  /** Returns a string representation of the object.
   * \return A string representation of the object.
   */
  string toString(void) const
  {
    string str = "TaskApproximateQuadraticFunctionSolver";
    return str;
  }
  
private:
  VectorXd inputs_;
  VectorXd targets_;
  double regularization_weight_;
};


/** The task solver tunes the parameters a and c such that the function \f$ y = a*x^2 + c \f$ best matches a set of target values y_target for a set of input values x
 */
class DemoTaskSolverApproximateQuadraticFunction : public TaskSolver
{
public:
  /**
   *  \param[in] inputs x in \f$ y = a*x^2 + c \f$
   */
  DemoTaskSolverApproximateQuadraticFunction(const VectorXd& inputs) 
  {
    inputs_ = inputs;
  }
  
  /** Function to perform a rollout
   * \param[in] sample Sample containing variation of a and c  (in  \f$ y = a*x^2 + c \f$)
   * \param[in] task_parameters Ignored
   * \param[in] cost_vars Cost-relevant variables, containing the predictions
   */
  void performRollout(const VectorXd& sample, const VectorXd& task_parameters, MatrixXd& cost_vars) const 
  {
    
    VectorXd predictions;
    double a = sample(0);
    double c = sample(1);
    targetFunction(a,c,inputs_,predictions);
    cost_vars = predictions;
  }
  
  /** Returns a string representation of the object.
   * \return A string representation of the object.
   */
  string toString(void) const
  {
    string str = "TaskApproximateQuadraticFunctionSolver";
    return str;
  }
  
private:
  VectorXd inputs_;
};




/** Main function
 * \param[in] n_args Number of arguments
 * \param[in] args Arguments themselves
 * \return Success of exection. 0 if successful.
 */
int main(int n_args, char* args[])
{
  // If program has an argument, it is a directory to which to save files too (or --help)
  string directory;
  string covar_update;
  if (n_args>1)
  {
    if (string(args[1]).compare("--help")==0)
    {
      cout << "Usage: " << args[0] << " [directory] [covar_update]" << endl;
      cout << "                    directory: optional directory to save data to" << endl;
      cout << "                    covar_update: [none|decay|adaptation]" << endl;
      return 0;
    }
    else
    {
      directory = string(args[1]);
    }
    if (n_args>2)
    {
      covar_update = string(args[2]);
    }
  }
  
  VectorXd inputs = VectorXd::LinSpaced(21,-1.5,1.5);
  double a = 2.0;
  double c = -1.0;
  int n_params = 2;
  
  double regularization = 0.01;
  
  Task* task = new DemoTaskApproximateQuadraticFunction(a,c,inputs,regularization);
  TaskSolver* task_solver = new DemoTaskSolverApproximateQuadraticFunction(inputs); 
  
  VectorXd mean_init  =  0.5*VectorXd::Ones(n_params);
  MatrixXd covar_init =  0.25*MatrixXd::Identity(n_params,n_params);
  DistributionGaussian* distribution = new DistributionGaussian(mean_init, covar_init); 
  
  Updater* updater = NULL;
  double eliteness = 10;
  string weighting_method = "PI-BB";
    
  if (covar_update.compare("none")==0)
  {
    updater = new UpdaterMean(eliteness,weighting_method);
  }
  else if (covar_update.compare("decay")==0)
  {
    double covar_decay_factor = 0.9;
    updater = new UpdaterCovarDecay(eliteness,covar_decay_factor,weighting_method);
  }
  else 
  {
    VectorXd base_level = VectorXd::Constant(n_params,0.001);
    eliteness = 10;
    bool diag_only = false;
    double learning_rate = 0.5;
    updater = new UpdaterCovarAdaptation(eliteness,weighting_method,base_level,diag_only,learning_rate);  
  }

  
  int n_samples_per_update = 15;
  
  int n_updates = 40;

  // Here's one way to call it. Below there's another one using ExperimentBBO.
  //cout << "___________________________________________________________" << endl;
  //cout << "RUNNING OPTIMIZATION" << endl;  
  runOptimizationTask(task, task_solver, distribution, updater, n_updates, n_samples_per_update,directory);
  
  ExperimentBBO experiment(
    task,
    task_solver,
    distribution,
    updater,
    n_updates,
    n_samples_per_update
  );
  
  //cout << "___________________________________________________________" << endl;
  //cout << "RUNNING SAME OPTIMIZATION (WITH ExperimentBBO)" << endl;  
  //runOptimization(experiment, distribution, updater, n_updates, n_samples_per_update,directory);
  
  return 0;
}


