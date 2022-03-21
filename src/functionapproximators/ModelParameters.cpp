/**
 * @file   ModelParameters.cpp
 * @brief  ModelParameters class source file.
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
 
#include <iostream>
#include <assert.h>

#include "functionapproximators/ModelParameters.hpp"
#include "functionapproximators/UnifiedModel.hpp"

#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

namespace DmpBbo {

bool ModelParameters::isParameterSelected(std::string label) const {
   return selected_param_labels_.find(label)!=selected_param_labels_.end();
}

void ModelParameters::setSelectedParameters(const std::set<std::string>& labels)
{
  selected_param_labels_ = set<string>();
  
  // Check if all labels passed are actually possible
  set<string> possible_values_labels;
  getSelectableParameters(possible_values_labels);
  set<string>::iterator it;
  for (it = labels.begin(); it != labels.end(); ++it) {
    if (possible_values_labels.count(*it)==0) {
      cout << "WARNING: '" << *it << "' is an unknown label in Parameterizable." << endl;
    } else {
      selected_param_labels_.insert(*it);
    }
  }
  
}

int ModelParameters::getParameterVectorSize(void) const 
{
  int size = 0;
  for (const string& label: selected_param_labels_)
    size += sizes_.at(label);
  return size;
}

void ModelParameters::checkMinMax(void) {
  set<string> labels;
  getSelectableParameters(labels);
  for(auto label : labels) {
    if (min_[label] == max_[label]) {
        max_[label] = min_[label]+1.0;
    }
  }  
}

bool ModelParameters::saveGridData(const VectorXd& min, const VectorXd& max, const VectorXi& n_samples_per_dim, string save_directory, bool overwrite) const
{

  if (save_directory.empty())
    return true;

  UnifiedModel* mp_unified = toUnifiedModel();
  if (mp_unified==NULL)
    return false;

  return mp_unified->saveGridData(min,max,n_samples_per_dim,save_directory,overwrite);
  
}



}
