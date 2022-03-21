/**
 * @file   TaskSolverDmp.cpp
 * @brief  TaskSolverDmp class source file.
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

#include "dmp_bbo/TaskSolverDmp.hpp"

#include <iostream>
#include <string>
#include <set>
#include <eigen3/Eigen/Core>

#include "dmpbbo_io/EigenFileIO.hpp"
#include "dmp/Dmp.hpp"
#include "dmp/Trajectory.hpp"

using namespace std;
using namespace Eigen;

namespace DmpBbo {
  
TaskSolverDmp::TaskSolverDmp(Dmp* dmp, std::set<std::string> optimize_parameters, double dt, double integrate_dmp_beyond_tau_factor, bool use_normalized_parameter)
: dmp_(dmp)
{
  dmp_->setSelectedParameters(optimize_parameters);
  
  integrate_time_ = dmp_->tau() * integrate_dmp_beyond_tau_factor;
  n_time_steps_ = (integrate_time_/dt)+1;
  use_normalized_parameter_ = use_normalized_parameter;
}

void TaskSolverDmp::set_perturbation(double perturbation_standard_deviation)
{
  dmp_->set_perturbation_analytical_solution(perturbation_standard_deviation);
}

void TaskSolverDmp::performRollout(const Eigen::VectorXd& sample, const Eigen::VectorXd& task_parameters, Eigen::MatrixXd& cost_vars) const
{

  dmp_->setParameterVector(sample, use_normalized_parameter_);

  Trajectory traj;
  MatrixXd forcing_terms;
  VectorXd ts = VectorXd::LinSpaced(n_time_steps_,0.0,integrate_time_);
  dmp_->analyticalSolution(ts,traj,forcing_terms);
  traj.set_misc(forcing_terms);
  
  traj.asMatrix(cost_vars); 
}


string TaskSolverDmp::toString(void) const
{
  return string("TaskSolverDmp");
}

}
