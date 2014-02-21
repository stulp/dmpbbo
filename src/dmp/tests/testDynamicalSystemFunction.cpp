/**
 * \file testDynamicalSystemFunction.cpp
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


#include "dynamicalsystems/DynamicalSystem.hpp"
#include "testDynamicalSystemFunction.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

void testDynamicalSystemFunction(DynamicalSystem* dyn_sys, double dt, int T, string directory)
{
  VectorXd x(dyn_sys->dim(),1);
  VectorXd xd(dyn_sys->dim(),1);
  VectorXd x_updated(dyn_sys->dim(),1);
  dyn_sys->integrateStart(x,xd);

  MatrixXd xs_step(T,x.size());
  MatrixXd xds_step(T,xd.size());
  xs_step.row(0) = x;
  xds_step.row(0) = xd;
  
  cout << "** Integrate step-by-step." << endl;
  VectorXd ts = VectorXd::Zero(T);
  for (int t=1; t<T; t++)
  {
    dyn_sys->integrateStep(dt,x,x_updated,xd); 
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
  dyn_sys->analyticalSolution(ts,xs_ana,xds_ana);
  
  if (!directory.empty())
  {
    cout << "** Write data." << endl;

    bool overwrite=true;
    
    MatrixXd output_ana(T,1+xs_ana.cols()+xds_ana.cols());
    output_ana << xs_ana, xds_ana, ts;
    saveMatrix(directory,"analytical.txt",output_ana,overwrite);

    MatrixXd output_step(T,1+xs_ana.cols()+xds_ana.cols());
    output_step << xs_step, xds_step, ts;
    saveMatrix(directory,"step.txt",output_step,overwrite);

    MatrixXd tau_mat(1,1);
    tau_mat(0,0) = dyn_sys->tau();
    saveMatrix(directory,"tau.txt",tau_mat,overwrite);
  }
}

