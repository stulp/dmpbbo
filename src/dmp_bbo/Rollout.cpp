/**
 * @file   Rollout.cpp
 * @brief  Rollout class source file.
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

#include "dmp_bbo/Rollout.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

namespace DmpBbo {

  
Rollout::Rollout(const Eigen::MatrixXd& policy_parameters):
  policy_parameters_(policy_parameters),
  cost_vars_(MatrixXd(0,0)),
  cost_(VectorXd(0,0))
{
}
  
Rollout::Rollout(const Eigen::MatrixXd& policy_parameters, const Eigen::MatrixXd& cost_vars):
  policy_parameters_(policy_parameters),
  cost_vars_(cost_vars),
  cost_(VectorXd(0,0))
{
}

Rollout::Rollout(const Eigen::MatrixXd& policy_parameters, const Eigen::MatrixXd& cost_vars, const Eigen::MatrixXd& cost):
  policy_parameters_(policy_parameters),
  cost_vars_(cost_vars),
  cost_(cost)
{
}

unsigned int Rollout::getNumberOfCostComponents(void) const
{
  if (cost_.size()==0)
  {
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "This rollout has not cost yet!" << std::endl;
    return 0;
  }

  return cost_.size();
}

double Rollout::total_cost(void) const
{ 
  if (cost_.size()==0)
  {
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "This rollout has not cost yet!" << std::endl;
    return 0.0;
  }
  return cost_[0];
}


bool Rollout::saveToDirectory(std::string directory, bool overwrite) const
{
  // Make directory if it doesn't already exist
  if (!boost::filesystem::exists(directory))
  {
    if (!boost::filesystem::create_directories(directory))
    {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "Couldn't make directory file '" << directory << "'." << endl;
      return false;
    }
  }

  // Abbreviations to make it fit on one line
  bool ow = overwrite;
  string dir = directory;
  
  if (!saveMatrix(dir, "policy_parameters.txt", policy_parameters_,  ow)) return false;
  if (cost_vars_.size()>0)
    if (!saveMatrix(dir, "cost_vars.txt",       cost_vars_,          ow)) return false;
  if (cost_.size()>0)
    if (!saveMatrix(dir, "cost.txt",            cost_,               ow)) return false;
  
  return true;
  
}


}
