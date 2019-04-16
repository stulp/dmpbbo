/**
 * @file   TaskSolver.cpp
 * @brief  TaskSolver class source file.
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

#include "dmp_bbo/TaskSolver.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

namespace DmpBbo {
  
TaskSolver::TaskSolver(void)
{
}

TaskSolver::~TaskSolver(void)
{
}

void TaskSolver::performRollout(const Eigen::VectorXd& sample, Eigen::MatrixXd& cost_vars) const
{
  Eigen::VectorXd task_parameters;
  performRollout(sample,task_parameters,cost_vars);
};


}
