/**
 * \file testPerturbModelParametersLWR.cpp
 * \author Freek Stulp
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
#include <iomanip>
#include <string>
#include <fstream>

#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/ModelParametersLWR.hpp"
#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "../demos/targetFunction.hpp"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

int main(int n_args, char** args)
{
  string directory;
  if (n_args>1)
    directory = string(args[1]);
  //else
  //  usage(args[0],"/tmp/testFunctionApproximatorLWR");

  for (int input_dim=1; input_dim<=2; input_dim++)
  {
    cout << "________________________________________________________________________" << endl;
    cout << "________________________________________________________________________" << endl;

    VectorXi n_samples_per_dim = VectorXi::Constant(1,10);
    if (input_dim==2) 
      n_samples_per_dim = VectorXi::Constant(2,25);
    
    MatrixXd inputs, targets, outputs;
    targetFunction(n_samples_per_dim,inputs,targets);


    double intersection = 0.5;
    int n_rfs = 9;
    if (input_dim==2) 
      n_rfs = 3;
      
    VectorXi num_rfs_per_dim = VectorXi::Constant(input_dim,n_rfs);
    MetaParametersLWR* meta_parameters = new MetaParametersLWR(input_dim,num_rfs_per_dim,intersection);

    string save_directory;
    if (!directory.empty())
      save_directory = directory+"/"+(input_dim==1?"1D":"2D");
    
    
    FunctionApproximator* fa = new FunctionApproximatorLWR(meta_parameters);
    bool overwrite = true;
    fa->train(inputs,targets,save_directory,overwrite);

    // Now the basic functionality of the LWR FA has been tested.
    // No perturb the model parameters
    const ModelParametersLWR* model_parameters_lwr_const = static_cast< const ModelParametersLWR*>(fa->getModelParameters());
    
    // Get a clone which is not const so that we can modify it
    ModelParametersLWR* model_parameters_lwr = static_cast< ModelParametersLWR*>(model_parameters_lwr_const->clone());
      
    set<string> selected;
    selected.insert("offsets");
    selected.insert("slopes");
    model_parameters_lwr->setSelectedParameters(selected);
    model_parameters_lwr->set_lines_pivot_at_max_activation(true);

    VectorXd values;
    bool normalized = false;
    model_parameters_lwr->getParameterVectorSelected(values,normalized);
    cout << "Original values             : " << fixed << setprecision(4) << values.transpose() << endl;
    normalized = true;
    model_parameters_lwr->getParameterVectorSelected(values,normalized);
    cout << "Original values (normalized): " << fixed << setprecision(4) << values.transpose() << endl;
    
    normalized = true;
    model_parameters_lwr->getParameterVectorSelected(values,normalized);
    
    int n_perturbations = 5;
    for (int i_perturbation=0; i_perturbation<n_perturbations; i_perturbation++)
    {
      if (!save_directory.empty())
      {
        // Get min and max of the targetfunction (i.e. generate just 2 samples per dimension)
        MatrixXd inputs;
        MatrixXd targets;
        targetFunction(VectorXi::Constant(input_dim,2), inputs, targets);
        VectorXd min = inputs.colwise().minCoeff();
        VectorXd max = inputs.colwise().maxCoeff();
      
        VectorXi n_samples_per_dim = VectorXi::Constant(input_dim,100);
        if (input_dim==2)
          n_samples_per_dim = VectorXi::Constant(input_dim,40);
          
        string cur_save_directory = save_directory + "/perturbation" + to_string(i_perturbation);
        
        FunctionApproximatorLWR fa(model_parameters_lwr);
        fa.saveGridData(min, max, n_samples_per_dim, cur_save_directory, overwrite);
      }

      double scale = 0.05;
      VectorXd perturbations = scale*VectorXd::Random(values.size());
      VectorXd values_perturbed = values.array()+perturbations.array();

      model_parameters_lwr->setParameterVectorSelected(values_perturbed,normalized);
      //cout << *model_parameters_lwr << endl;
      cout << "Perturbation " << i_perturbation << ": " << fixed << setprecision(4) << values_perturbed.transpose() << endl;
        
    }
    
    delete meta_parameters;
    delete fa;
  }
  
  
  
  
  return 0;
}

