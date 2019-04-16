/**
 * @file   TaskSolverDmpArm2D.cpp
 * @brief  TaskSolverDmpArm2D class source file.
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

#include "TaskSolverDmpArm2D.hpp"

#include <iostream>
#include <string>
#include <set>
#include <eigen3/Eigen/Core>

#include "dmp/Dmp.hpp"
#include "dmp/Trajectory.hpp"

using namespace std;
using namespace Eigen;

namespace DmpBbo {
  
TaskSolverDmpArm2D::TaskSolverDmpArm2D(Dmp* dmp, const Eigen::VectorXd& link_lengths, std::set<std::string> optimize_parameters, double dt, double integrate_dmp_beyond_tau_factor, bool use_normalized_parameter)
: TaskSolverDmp(dmp,optimize_parameters,dt,integrate_dmp_beyond_tau_factor,use_normalized_parameter), link_lengths_(link_lengths)
{
  // Every dimension of the DMP controls one joint
  // There is one joint between each pair of links
  int n_dofs = dmp->dim_orig();
  //assert( n_dofs == (link_lengths_.size()-1) ); // fff
  
  VectorXd ts = VectorXd::LinSpaced(n_time_steps_,0.0,integrate_time_);
  Trajectory traj_reproduced;
  dmp_->analyticalSolution(ts,traj_reproduced);
  traj_reproduced.saveToFile("/tmp/demoOptimizationDmpArm2D/3D/","traj_before.txt");

  VectorXd initial_angles;
  TaskSolverDmpArm2D::getInitialAngles(n_dofs,initial_angles);
  dmp_->set_initial_state(initial_angles);
  
  VectorXd final_angles;
  TaskSolverDmpArm2D::getFinalAngles(n_dofs,final_angles);
  dmp_->set_attractor_state(final_angles);
  
  dmp_->analyticalSolution(ts,traj_reproduced);
  traj_reproduced.saveToFile("/tmp/demoOptimizationDmpArm2D/3D/","traj_after.txt");
}

void TaskSolverDmpArm2D::getInitialAngles(unsigned int n_dofs, VectorXd& initial_angles) {
  initial_angles = VectorXd::Zero(n_dofs);
}
  
void TaskSolverDmpArm2D::getFinalAngles(unsigned int n_dofs, VectorXd& final_angles) {
  final_angles = VectorXd::Constant(n_dofs,M_PI/n_dofs);
  final_angles[0] = final_angles[0]/2.0;
}


void TaskSolverDmpArm2D::performRollout(const Eigen::VectorXd& sample, const Eigen::VectorXd& task_parameters, Eigen::MatrixXd& cost_vars) const
{
  TaskSolverDmp::performRollout(sample, task_parameters, cost_vars);

  // cost_vars_angles structure is (without the link positions!)
  //
  // time  joint angles (e.g. n_dofs = 3)     forcing term  link positions (e.g. 3+1) 
  // ____  __________________________________  __________  _________________________    
  // t     | a a a | ad ad ad | add add add |  f f f       | x y | x y | x y | x y  |    
  //
  // We now compute the link positions and add them to the end
  
  int n_dofs = dmp_->dim_orig();
  int n_time_steps = cost_vars.rows();
  assert(cost_vars.cols() == 1+4*n_dofs);
  
  // Make room for link positions, i.e. 2 * (n_links+1)
  cost_vars.conservativeResize(n_time_steps, 1 + 4*n_dofs + 2*(n_dofs+1));
  
  MatrixXd angles = cost_vars.block(0,1,n_time_steps,n_dofs);
  MatrixXd link_positions;
  anglesToLinkPositions(angles,link_positions);
  
  // Add the link positions to the right side of the cost_vars matrix
  cost_vars.rightCols(2*(n_dofs+1)) = link_positions;
}

void TaskSolverDmpArm2D::anglesToLinkPositions(const MatrixXd& angles, MatrixXd& link_positions) const
{
  int n_time_steps = angles.rows();
  int n_dofs = angles.cols();
  MatrixXd links_x = MatrixXd::Zero(n_time_steps_,n_dofs+1);
  MatrixXd links_y = MatrixXd::Zero(n_time_steps_,n_dofs+1);
  for (int tt=0; tt<n_time_steps_; tt++)
  {
    double sum_angles = 0.0;
    for (int i_dof=0; i_dof<n_dofs; i_dof++)
    {
      sum_angles = sum_angles + angles(tt,i_dof);
      links_x(tt,i_dof+1) = links_x(tt,i_dof) + cos(sum_angles)*link_lengths_(i_dof);
      links_y(tt,i_dof+1) = links_y(tt,i_dof) + sin(sum_angles)*link_lengths_(i_dof);
    }
  }
  
  // x y x y x y
  link_positions.resize(n_time_steps,2*(n_dofs+1));
  for (int i_link=0; i_link<n_dofs+1; i_link++)
  {
    link_positions.col(2*i_link+0) = links_x.col(i_link);
    link_positions.col(2*i_link+1) = links_y.col(i_link);
  }
}


string TaskSolverDmpArm2D::toString(void) const
{
  return string("TaskSolverDmpArm2D");
}

}
