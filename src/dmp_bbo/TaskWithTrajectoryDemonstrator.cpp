/**
 * @file   TaskWithTrajectoryDemonstrator.cpp
 * @brief  TaskWithTrajectoryDemonstrator class source file.
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

#include "dmp_bbo/TaskWithTrajectoryDemonstrator.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

namespace DmpBbo {

TaskWithTrajectoryDemonstrator::TaskWithTrajectoryDemonstrator(void)
{
}


TaskWithTrajectoryDemonstrator::~TaskWithTrajectoryDemonstrator(void)
{
}

void TaskWithTrajectoryDemonstrator::generateDemonstrations(const vector<MatrixXd>& task_parameters, const vector<VectorXd>& ts, vector<Trajectory>& demonstrations) const
{
  unsigned int n_demos = task_parameters.size();
  assert(n_demos==ts.size());
  
  demonstrations = vector<Trajectory>(n_demos);
  for (unsigned int i_demo=0; i_demo<n_demos; i_demo++)
  {
    generateDemonstration(task_parameters[i_demo], ts[i_demo], demonstrations[i_demo]); 
  }
  
}


void TaskWithTrajectoryDemonstrator::generateDemonstrations(DistributionGaussian* task_parameter_distribution, int n_demos, const VectorXd& ts, vector<Trajectory>& demonstrations) const
{
  MatrixXd task_parameters;
  
  demonstrations = vector<Trajectory>(n_demos);
  for (int i_demo=0; i_demo<n_demos; i_demo++)
  {
    task_parameter_distribution->generateSamples(n_demos,task_parameters);
    generateDemonstration(task_parameters, ts, demonstrations[i_demo]); 
  }
  
}




}
