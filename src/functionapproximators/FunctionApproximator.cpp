/**
 * @file   FunctionApproximator.cpp
 * @brief  FunctionApproximator class source file.
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

#include "functionapproximators/FunctionApproximator.hpp"

#include "functionapproximators/ModelParameters.hpp"

#include "eigen/eigen_file_io.hpp"

#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Core>
#include <boost/filesystem.hpp> // Required only for train(inputs,outputs,save_directory)

using namespace std;
using namespace Eigen;


/** \ingroup FunctionApproximators
 */

namespace DmpBbo { 

FunctionApproximator::FunctionApproximator(const ModelParameters *const model_parameters) 
{
  assert(model_parameters!=NULL);
  model_parameters_ = model_parameters->clone();
}

FunctionApproximator::~FunctionApproximator(void) 
{
  delete model_parameters_;
}

  
/******************************************************************************/
const ModelParameters* FunctionApproximator::getModelParameters(void) const 
{ 
  return model_parameters_; 
};

/******************************************************************************/
void FunctionApproximator::setModelParameters(ModelParameters* model_parameters)
{
  if (model_parameters_!=NULL)
  {
    delete model_parameters_;
    model_parameters_ = NULL;
  }

  model_parameters_ = model_parameters;
}

int FunctionApproximator::getExpectedInputDim(void) const
{
  return model_parameters_->getExpectedInputDim();
}

int FunctionApproximator::getExpectedOutputDim(void) const
{
  return model_parameters_->getExpectedOutputDim();
}


void FunctionApproximator::getParameterVectorSelectedMinMax(Eigen::VectorXd& min, Eigen::VectorXd& max) const
{
  if (model_parameters_==NULL)
  {
    cerr << __FILE__ << ":" << __LINE__ << ": Warning: Trying to access model parameters of the function approximator, but it has not been trained yet. Returning empty parameter vector." << endl;
    min.resize(0);
    max.resize(0);
    return;
  }

  model_parameters_->getParameterVectorSelectedMinMax(min,max);
}

/******************************************************************************/
bool FunctionApproximator::checkModelParametersInitialized(void) const
{
  if (model_parameters_==NULL)
  {
    cerr << "Warning: Trying to access model parameters of the function approximator, but it has not been trained yet. Returning empty parameter vector." << endl;
    return false;
  }
  return true;
  
}

/******************************************************************************/
void FunctionApproximator::getParameterVectorSelected(VectorXd& values, bool normalized) const
{
  if (checkModelParametersInitialized())
    model_parameters_->getParameterVectorSelected(values, normalized);
  else
    values.resize(0);
}

/******************************************************************************/
int FunctionApproximator::getParameterVectorSelectedSize(void) const
{
  if (checkModelParametersInitialized())
    return model_parameters_->getParameterVectorSelectedSize();
  else 
    return 0; 
}

void FunctionApproximator::setParameterVectorSelected(const VectorXd& values, bool normalized) 
{
  if (checkModelParametersInitialized())
    model_parameters_->setParameterVectorSelected(values, normalized);
}

void FunctionApproximator::setSelectedParameters(const set<string>& selected_values_labels)
{
  if (checkModelParametersInitialized())
    model_parameters_->setSelectedParameters(selected_values_labels);
}

void FunctionApproximator::getSelectableParameters(set<string>& labels) const
{
  if (checkModelParametersInitialized())
    model_parameters_->getSelectableParameters(labels);
  else
    labels.clear();
}

void FunctionApproximator::getParameterVectorMask(const std::set<std::string> selected_values_labels, Eigen::VectorXi& selected_mask) const {
  if (checkModelParametersInitialized())
    model_parameters_->getParameterVectorMask(selected_values_labels,selected_mask);
  else
    selected_mask.resize(0);
  
};
int FunctionApproximator::getParameterVectorAllSize(void) const {
  if (checkModelParametersInitialized())
    return model_parameters_->getParameterVectorAllSize();
  else
    return 0;
};
void FunctionApproximator::getParameterVectorAll(Eigen::VectorXd& values) const {
  if (checkModelParametersInitialized())
    model_parameters_->getParameterVectorAll(values);    
  else
    values.resize(0);
};
void FunctionApproximator::setParameterVectorAll(const Eigen::VectorXd& values) {
  if (checkModelParametersInitialized())
    model_parameters_->setParameterVectorAll(values);
};


string FunctionApproximator::toString(void) const
{
  std::stringstream s;
  s << "FunctionApproximator"+getName() << endl;
  if (model_parameters_!=NULL)
    s << *model_parameters_ << endl;
  return s.str();
}

void FunctionApproximator::setParameterVectorModifierPrivate(std::string modifier, bool new_value)
{
  model_parameters_->setParameterVectorModifier(modifier,new_value);
}

}

