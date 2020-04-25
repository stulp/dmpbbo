/**
 * @file   TaskViapointArm2D.cpp
 * @brief  TaskViapointArm2D class source file.
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

#include "dmp_bbo/tasks/TaskViapointArm2D.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <eigen3/Eigen/Core>


using namespace std;
using namespace Eigen;

namespace DmpBbo {

TaskViapointArm2D::TaskViapointArm2D(int n_dofs, const Eigen::VectorXd& viapoint, double  viapoint_time, double viapoint_radius)
: TaskViapoint(viapoint, viapoint_time,viapoint_radius), n_dofs_(n_dofs)
{
  assert(viapoint.size()==2); // Because TaskViapointArm2D => 2D
}


void TaskViapointArm2D::evaluateRollout(const MatrixXd& cost_vars, const Eigen::VectorXd& sample, const VectorXd& task_parameters, VectorXd& costs) const
{
  int n_time_steps = cost_vars.rows();
  //int n_cost_vars = cost_vars.cols();
  
  // cost_vars is assumed to have following structure
  // time  joint angles (e.g. n_dofs = 3)     forcing term  link positions (e.g. 3+1) 
  // ____  __________________________________  __________  __________________________    
  // t     | a a a | ad ad ad | add add add |  f f f       | x y | x y | x y | x y  |     
  //
  // 1     + 3*n_dofs                        + n_dofs +       2*(n_dofs+1))
  //
  // Thus, the following must be true: n_cost_vars =  1 + 3*n_dofs + n_dofs + 2*(n_dofs+1)
  assert(cost_vars.cols() == 1 + 3*n_dofs_ + n_dofs_ + 2*(n_dofs_+1));
  
  // rollout is of size   n_time_steps x n_cost_vars
  VectorXd ts = cost_vars.col(0);
  // Link positions of end-effector, they are on the far right in the matrix, and always 2D
  MatrixXd y = cost_vars.rightCols(2);
  // Joint accelerations
  MatrixXd add = cost_vars.block(0,1+2*n_dofs_,n_time_steps,n_dofs_);
  computeCosts(ts,y,add,costs);         
}


string TaskViapointArm2D::toString(void) const {
  return string("TaskViapointArm2D");
}



}
