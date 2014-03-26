/**
 * \file testDmpModelParameterValues.cpp
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

#include "dmp/Dmp.hpp"

#include "dynamicalsystems/DynamicalSystem.hpp"
#include "dynamicalsystems/ExponentialSystem.hpp"
#include "dynamicalsystems/SigmoidSystem.hpp"
#include "dynamicalsystems/TimeSystem.hpp"
#include "dynamicalsystems/SpringDamperSystem.hpp"

#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/ModelParametersLWR.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"

#include <eigen3/Eigen/Core>
#include <boost/filesystem.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

void testDmp(Dmp* dmp, double dt, int T, std::string directory="");


int main(int n_args, char** args)
{
  Dmp* dmp = NULL; 

  // Some default values for dynamical system
  double tau = 0.6; 
  int dim = 2;
  VectorXd y_init(dim); 
  y_init   << -100.0 , -200.0; 
  VectorXd y_attr(dim);
  y_attr << 100.0, 200.0; 

  Dmp::DmpType dmp_type = Dmp::KULVICIUS_2012_JOINING;
  int input_dim = 1;
  
  
  vector<FunctionApproximator*> function_approximators(dim);    
  // Initialize one LWR for each dimension
  VectorXi n_basis_functions_vector(2); n_basis_functions_vector << 2, 3; 
  for (int dd=0; dd<dim; dd++)
  {
    int n_basis_functions = n_basis_functions_vector[dd];
    VectorXd centers = VectorXd::LinSpaced(n_basis_functions,0,1);
    VectorXd widths  = VectorXd::Constant(n_basis_functions,centers[1]-centers[0]);
    VectorXd slopes = VectorXd::Constant(n_basis_functions,(dd+1)*0.2);
    VectorXd offsets = VectorXd::LinSpaced(n_basis_functions,10*(dd+1),16*(dd+1));
    
    MetaParametersLWR* meta_parameters = new MetaParametersLWR(input_dim,n_basis_functions);      
    ModelParametersLWR* model_parameters = new ModelParametersLWR(centers,widths,slopes,offsets);
    function_approximators[dd] = new FunctionApproximatorLWR(meta_parameters,model_parameters);
  }
  
  dmp = new Dmp(tau, y_init, y_attr, function_approximators, dmp_type);
  dmp->set_name("testDmp");

  cout << *dmp << endl;
  
  set<string> selected_labels;  
  selected_labels.insert("offsets");
  selected_labels.insert("slopes");
  //selected_labels.insert("widths");
  //selected_labels.insert("centers");

  selected_labels.insert("goal");
  
  dmp->setSelectedParameters(selected_labels);
  
  cout << "vector size (all     ) = " << dmp->getParameterVectorAllSize() << endl;
  cout << "vector size (selected) = " <<  dmp->getParameterVectorSelectedSize() << endl;
  
  VectorXi selected_mask;
  VectorXd values, min_values, max_values, values_normalized;


  dmp->getParameterVectorMask(selected_labels,selected_mask);
  cout << "mask = " << selected_mask.transpose() << endl << endl;
  
  dmp->getParameterVectorAll(values);
  cout << "values     (all     ): " << values.transpose() << endl;
  
  dmp->getParameterVectorSelected(values);
  cout << "values     (selected): " << values.transpose() << endl;
  
  dmp->getParameterVectorSelectedMinMax(min_values,max_values);
  cout << "min_values (selected): " << min_values.transpose() << endl;
  cout << "max_values (selected): " << max_values.transpose() << endl ;
  
  dmp->getParameterVectorSelectedNormalized(values_normalized);
  cout << "values_norm(selected): " << values_normalized.transpose() << endl << endl;

  cout << "_______________________________________________" << endl;
  
  VectorXd new_values = 3*values;
  //VectorXd new_values = VectorXd::LinSpaced(dmp->getParameterVectorSelectedSize(),100,210);
  dmp->setParameterVectorSelected(new_values);
  
  dmp->getParameterVectorAll(values);
  cout << "values     (all     ): " << values.transpose() << endl;
  
  dmp->getParameterVectorSelected(values);
  cout << "values     (selected): " << values.transpose() << endl;
  
  dmp->getParameterVectorSelectedMinMax(min_values,max_values);
  cout << "min_values (selected): " << min_values.transpose() << endl;
  cout << "max_values (selected): " << max_values.transpose() << endl ;
  
  dmp->getParameterVectorSelectedNormalized(values_normalized);
  cout << "values_norm(selected): " << values_normalized.transpose() << endl << endl;

  cout << "_______________________________________________" << endl;
  vector<VectorXd> vector_values(dim);
  int goal_space = 0;
  if (selected_labels.find("goal")!=selected_labels.end())
    goal_space = 1;
  cout << "  goal_space=" << goal_space << endl;
  vector_values[0] = VectorXd::Constant(2*n_basis_functions_vector[0]+goal_space,-1);
  vector_values[1] = VectorXd::Constant(2*n_basis_functions_vector[1]+goal_space,-1);
  
  dmp->setParameterVectorSelected(vector_values);
  
  dmp->getParameterVectorAll(values);
  cout << "values     (all     ): " << values.transpose() << endl;
  
  dmp->getParameterVectorSelected(values);
  cout << "values     (selected): " << values.transpose() << endl;
  
  dmp->getParameterVectorSelectedMinMax(min_values,max_values);
  cout << "min_values (selected): " << min_values.transpose() << endl;
  cout << "max_values (selected): " << max_values.transpose() << endl ;
  
  dmp->getParameterVectorSelectedNormalized(values_normalized);
  cout << "values_norm(selected): " << values_normalized.transpose() << endl << endl;
  
  delete dmp;
 
  return 0;

}


