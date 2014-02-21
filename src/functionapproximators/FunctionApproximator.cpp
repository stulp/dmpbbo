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
#include "functionapproximators/MetaParameters.hpp"

#include "utilities/EigenFileIO.hpp"
#include "utilities/EigenBoostSerialization.hpp"
#include "utilities/BoostSerializationToString.hpp"

#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Core>
#include <boost/filesystem.hpp> // Required only for train(inputs,outputs,save_directory)

using namespace std;
using namespace Eigen;


/** \defgroup FunctionApproximators Function Approximators
 */

namespace DmpBbo { 

/******************************************************************************/
FunctionApproximator::FunctionApproximator(MetaParameters *meta_parameters, ModelParameters *model_parameters) 
{
  assert(meta_parameters!=NULL);
  
  meta_parameters_  = meta_parameters->clone();
  if (model_parameters!=NULL)
  {
    model_parameters_ = model_parameters->clone();
    assert(model_parameters_->getExpectedInputDim()==meta_parameters_->getExpectedInputDim());
  }
  else
  {
    model_parameters_ = NULL;
  }

}

FunctionApproximator::FunctionApproximator(ModelParameters *model_parameters) 
{
  assert(model_parameters!=NULL);
  meta_parameters_  = NULL;
  model_parameters_ = model_parameters->clone();
}

FunctionApproximator::~FunctionApproximator(void) 
{
  delete meta_parameters_;
  delete model_parameters_;
}


/******************************************************************************/
const MetaParameters* FunctionApproximator::getMetaParameters(void) const 
{ 
  return meta_parameters_; 
};
  
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
  if (model_parameters_!=NULL)
    return model_parameters_->getExpectedInputDim();
  else
    return meta_parameters_->getExpectedInputDim();
}

void FunctionApproximator::reTrain(const MatrixXd& inputs, const MatrixXd& targets)
{
  delete model_parameters_;
  model_parameters_ = NULL;
  train(inputs,targets);
}

void FunctionApproximator::reTrain(const MatrixXd& inputs, const MatrixXd& targets, string save_directory, bool overwrite)
{
  delete model_parameters_;
  model_parameters_ = NULL;
  train(inputs,targets,save_directory,overwrite);
}


void FunctionApproximator::getParameterVectorSelectedMinMax(VectorXd& min, VectorXd& max) const
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
  string name = "FunctionApproximator"+getName();
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML(name.c_str());
}


void FunctionApproximator::train(const MatrixXd& inputs, const MatrixXd& targets, string save_directory, bool overwrite)
{
  train(inputs,targets);
  
  if (save_directory.empty())
    return;
  
  if (!isTrained())
    return;
  
  if (getExpectedInputDim()<3)
  {
    
    VectorXd min = inputs.colwise().minCoeff();
    VectorXd max = inputs.colwise().maxCoeff();
    
    int n_samples_per_dim = 100;
    if (getExpectedInputDim()==2) n_samples_per_dim = 40;
    VectorXi n_samples_per_dim_vec = VectorXi::Constant(getExpectedInputDim(),n_samples_per_dim);

    model_parameters_->saveGridData(min, max, n_samples_per_dim_vec, save_directory, overwrite);
    
  }

  MatrixXd outputs;
  predict(inputs,outputs);

  saveMatrix(save_directory,"inputs.txt",inputs,overwrite);
  saveMatrix(save_directory,"targets.txt",targets,overwrite);
  saveMatrix(save_directory,"outputs.txt",outputs,overwrite);
  
  string filename = save_directory+"/plotdata.py";
  ofstream outfile;
  outfile.open(filename.c_str()); 
  if (!outfile.is_open())
  {
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "Could not open file " << filename << " for writing." << endl;
  } 
  else
  {
    // Python code generation in C++. Rock 'n' roll! ;-)
    if (inputs.cols()==2) {                                                                                           
      outfile << "from mpl_toolkits.mplot3d import Axes3D                                       \n";
    }
    outfile   << "import numpy                                                                  \n";
    outfile   << "import matplotlib.pyplot as plt                                               \n";
    outfile   << "directory = '" << save_directory << "'                                        \n";
    outfile   << "inputs   = numpy.loadtxt(directory+'/inputs.txt')                             \n";
    outfile   << "targets  = numpy.loadtxt(directory+'/targets.txt')                            \n";
    outfile   << "outputs  = numpy.loadtxt(directory+'/outputs.txt')                            \n";
    outfile   << "fig = plt.figure()                                                            \n";
    if (inputs.cols()==2) {                                                                                           
      outfile << "ax = Axes3D(fig)                                                              \n";
      outfile << "ax.plot(inputs[:,0],inputs[:,1],targets, '.', label='targets',color='black')  \n";
      outfile << "ax.plot(inputs[:,0],inputs[:,1],outputs, '.', label='predictions',color='red')\n";
      outfile << "ax.set_xlabel('input_1'); ax.set_ylabel('input_2'); ax.set_zlabel('output')   \n";
      outfile << "ax.legend(loc='lower right')                                                  \n";
    } else {                                                                                           
      outfile << "plt.plot(inputs,targets, '.', label='targets',color='black')                  \n";
      outfile << "plt.plot(inputs,outputs, '.', label='predictions',color='red')                \n";
      outfile << "plt.xlabel('input'); plt.ylabel('output');                                    \n";
      outfile << "plt.legend(loc='lower right')                                                 \n";
    }                                                                                           
    outfile   << "plt.show()                                                                    \n";
    outfile << endl;

    outfile.close();
    //cout << "        ______________________________________________________________" << endl;
    //cout << "        | Plot saved data with:" << " 'python " << filename << "'." << endl;
    //cout << "        |______________________________________________________________" << endl;
  }
  
}

void FunctionApproximator::setParameterVectorModifierPrivate(std::string modifier, bool new_value)
{
  model_parameters_->setParameterVectorModifier(modifier,new_value);
}

}

