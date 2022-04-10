/**
 * \file testDmp.cpp
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
#include "functionapproximators/ModelParametersLWR.hpp"

#include "eigen/eigen_file_io.hpp"

#include <eigen3/Eigen/Core>
#include <boost/filesystem.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

void testDmp(Dmp* dmp, double dt, int T, std::string directory="");

void usage(string exec_name)
{
  cout << "Usage:" << endl;
  cout << "   " << exec_name << " [directory]" << endl;
  cout << "           Initializes default DMP" << endl;
  cout << "           Writes results to standard output or files (if passed as arguments)." << endl << endl;
  cout << "   " << exec_name << " --file <input_filename> [directory]" << endl;
  cout << "           Reads system from file 'input_filename'" << endl;
  cout << "           Writes results to standard output or files (if passed as arguments)." << endl;
}

int main(int n_args, char** args)
{
  string flag;
  string directory;
  
  if (n_args>1) {
    flag = string(args[1]);
    if (string("--help")==flag)
    {
      usage(args[0]);
      return 0;
    }
  }
    
  Dmp* dmp = NULL; 

  // Some default values for integration
  double dt = 0.004;
  int T = 250;

  // Some default values for dynamical system
  double tau = 0.6; 
  int dim = 3;
  VectorXd y_init(dim); 
  y_init   << 0.5, 0.4, 1.0; 
  VectorXd y_attr(dim);
  y_attr << 0.8, 0.1, 0.1; 

  if (string("--file")==flag)
  {
    if (n_args>2) {
      //   0     1          2                 3                            4
      // exec --file <input_filename> 
      // exec --file <input_filename> [output_filename_analytical]
      // exec --file <input_filename> [output_filename_analytical] [output_filename_step]
      
      /*
      string filename(args[2]);
      cout << "Reading DMP from file '" << filename << "'" << endl;
      ifstream file;
      file.open(filename.c_str());
      if (!file.is_open()) 
      {
        cerr << __FILE__ << ":" << __LINE__ << ": Can't find file '" << filename << "'. Abort." << endl; 
        return -1;
      }
  
      dmp = new Dmp();
      file >> *dmp;
      
      file.ignore(100,'=');
      file >> dt;
      file.ignore(100,'=');
      file >> T;
      
      file.close();
      */
      
      if (n_args>3)
        //   0     1          2                 3                            
        // exec --file <input_filename> [directory]
        directory = string(args[3]);
    } 
    else
    {
      usage(args[1]);
      return -1;
    }
  }
  else 
  {
    if (n_args>1)
      //   0         1                                      
      // exec [save_directory]
      directory = string(args[1]);
  }
    
  if (dmp==NULL)
  {
    Dmp::DmpType dmp_type = Dmp::KULVICIUS_2012_JOINING;
    dmp_type = Dmp::IJSPEERT_2002_MOVEMENT;
    
    //int input_dim = 1;
    
    int n_basis_functions = 9;
    
    // Dmp was not initialized above. Do so here now.
    vector<FunctionApproximator*> function_approximators(dim);    


    // Initialize one LWR for each dimension
    VectorXd offsets = VectorXd::Zero(n_basis_functions);
    
    MatrixXd slopes(dim,n_basis_functions);
    slopes << 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 10, 20, 100, -30, -120, 10, 10, -20,
              0, 0, 100, 0, 100, 0, 0, 0, 0;

              VectorXd centers = VectorXd::LinSpaced(n_basis_functions,0,1);
    VectorXd widths  = VectorXd::Constant(n_basis_functions,centers[1]-centers[0]);
    if (dmp_type==Dmp::IJSPEERT_2002_MOVEMENT)
    {
      centers = VectorXd::LinSpaced(n_basis_functions,0,tau);
      double alpha = 4.0;
      centers = (-alpha*(centers/tau)).array().exp();
      for (int ii=0; ii<(widths.size()-1); ii++)
        widths[ii]  = fabs(0.5*(centers[ii+1]-centers[ii]));
      widths[widths.size()-1] = widths[widths.size()-2];
      
      slopes = 3*slopes;
    }
              
    //double intersection_ratio = 0.5;
    for (int dd=0; dd<dim; dd++)
    {
      VectorXd cur_slopes = slopes.row(dd);
      ModelParametersLWR* model_parameters = new ModelParametersLWR(centers,widths,cur_slopes,offsets);
      function_approximators[dd] = new FunctionApproximatorLWR(model_parameters);
    }
    
    dmp = new Dmp(tau, y_init, y_attr, function_approximators, dmp_type);
  }

  VectorXd x(dmp->dim(),1);
  VectorXd xd(dmp->dim(),1);
  VectorXd x_updated(dmp->dim(),1);
  dmp->integrateStart(x,xd);

  MatrixXd xs_step(T,x.size());
  MatrixXd xds_step(T,xd.size());
  xs_step.row(0) = x;
  xds_step.row(0) = xd;
  
  cout << "** Integrate step-by-step." << endl;
  VectorXd ts = VectorXd::Zero(T);
  for (int t=1; t<T; t++)
  {
    dmp->integrateStep(dt,x,x_updated,xd); 
    x = x_updated;
    xs_step.row(t) = x;
    xds_step.row(t) = xd;
    if (directory.empty())
    {
      // Not writing to file, output on cout instead.
      cout << x.transpose() << " | " << xd.transpose() << endl;
    }
    
    ts(t) = t*dt;
  } 
  
  cout << "** Integrate analytically." << endl;
  MatrixXd xs_ana;
  MatrixXd xds_ana;
  MatrixXd forcing_terms_ana, fa_output_ana;
  dmp->analyticalSolution(ts,xs_ana,xds_ana,forcing_terms_ana,fa_output_ana);
  
  if (!directory.empty())
  {
    cout << "** Write data." << endl;

    bool overwrite=true;
    
    MatrixXd output_ana(T,1+xs_ana.cols()+xds_ana.cols());
    output_ana << xs_ana, xds_ana, ts;
    saveMatrix(directory,"analytical.txt",output_ana,overwrite);

    saveMatrix(directory,"forcing_terms_analytical.txt",forcing_terms_ana,overwrite);
    saveMatrix(directory,"fa_output_analytical.txt",fa_output_ana,overwrite);
  
    MatrixXd output_step(T,1+xs_ana.cols()+xds_ana.cols());
    output_step << xs_step, xds_step, ts;
    saveMatrix(directory,"step.txt",output_step,overwrite);

    MatrixXd tau_mat(1,1);
    tau_mat(0,0) = dmp->tau();
    saveMatrix(directory,"tau.txt",tau_mat,overwrite);
  }
  
  delete dmp;
 
  return 0;

}
