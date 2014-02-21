/**
 * @file   UpdateSummary.hpp
 * @brief  UpdateSummary class header file.
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

#include "bbo/UpdateSummary.hpp"

#include "bbo/DistributionGaussian.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

namespace DmpBbo {

bool saveToDirectory(const UpdateSummary& summary, string directory, int i_parallel)
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
  
  ofstream file;

  string suffix(".txt");
  if (i_parallel>0)
  {
    stringstream stream;
    stream << setw(2) << setfill('0') << i_parallel << ".txt";
    suffix = stream.str();
  }
  
  file.open((directory+"distribution_mean"+suffix).c_str());
  file << summary.distribution->mean();
  file.close();

  file.open((directory+"distribution_covar"+suffix).c_str());
  file << summary.distribution->covar();
  file.close();

  file.open((directory+"cost_eval"+suffix).c_str());
  file << summary.cost_eval;
  file.close();
  
  file.open((directory+"samples"+suffix).c_str());
  file << summary.samples;
  file.close();
  
  file.open((directory+"costs"+suffix).c_str());
  file << summary.costs;
  file.close();
  
  file.open((directory+"weights"+suffix).c_str());
  file << summary.weights;
  file.close();
  
  file.open((directory+"distribution_new_mean"+suffix).c_str());
  file << summary.distribution_new->mean();
  file.close();

  file.open((directory+"distribution_new_covar"+suffix).c_str());
  file << summary.distribution_new->covar();
  file.close();
  
  return true;
  
}

}
