/**
 * @file Trajectory.cpp
 * @brief  Trajectory class source file.
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

#include "dmp/Trajectory.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Core>


using namespace std;
using namespace Eigen;

namespace DmpBbo {
  
Trajectory::Trajectory(void)
{
}

Trajectory::Trajectory(const Eigen::VectorXd& ts, const Eigen::MatrixXd& ys,  const Eigen::MatrixXd& yds,  const Eigen::MatrixXd& ydds, const Eigen::MatrixXd& misc)
: ts_(ts), ys_(ys), yds_(yds), ydds_(ydds), misc_(misc)
{
  int n_time_steps = ts_.rows();
  assert(n_time_steps==ys_.rows());
  assert(n_time_steps==yds_.rows());
  assert(n_time_steps==ydds_.rows());
  if (misc_.cols()==0)
    misc_ = MatrixXd(n_time_steps,0);
  assert(n_time_steps==misc_.rows());
  
#ifndef NDEBUG // Variables below are only required for asserts; check for NDEBUG to avoid warnings.
  int n_dims = ys_.cols();
  assert(n_dims==yds_.cols());
  assert(n_dims==ydds_.cols());
#endif  
    
}

void Trajectory::set_misc(const Eigen::MatrixXd& misc)
{
  if (misc.rows()==ts_.size())
  {
    // misc is of size n_time_steps X n_dims_misc
    misc_ = misc;
  }
  else if (misc.rows()==1)
  {
    // misc is of size 1 X n_dim_misc
    // then replicate it so that it becomes of size n_time_steps X n_dims_misc
    misc_ = misc.replicate(ts_.size(),1);
  }
  else
  {
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "misc must have 1 or " << ts_.size() << " rows, but it has " << misc.rows() << endl;
  }
    
}

void Trajectory::append(const Trajectory& trajectory)
{
  assert(dim() == trajectory.dim());

  assert(ts_[length() - 1] == trajectory.ts()[0]);

  if (ys_.row(length() - 1).isZero() || trajectory.ys().row(0).isZero())
    assert(ys_.row(length() - 1).isZero() && trajectory.ys().row(0).isZero());
  else
    assert(ys_.row(length() - 1).isApprox(trajectory.ys().row(0)));

  if (yds_.row(length() - 1).isZero() || trajectory.yds().row(0).isZero())
    assert(yds_.row(length() - 1).isZero() && trajectory.yds().row(0).isZero());
  else
    assert(yds_.row(length() - 1).isApprox(trajectory.yds().row(0)));

  if (ydds_.row(length() - 1).isZero() || trajectory.ydds().row(0).isZero())
    assert(ydds_.row(length() - 1).isZero() && trajectory.ydds().row(0).isZero());
  else
    assert(ydds_.row(length() - 1).isApprox(trajectory.ydds().row(0)));

  int new_size = length() + trajectory.length() - 1;

  VectorXd new_ts(new_size);
  new_ts << ts_, trajectory.ts().segment(1, trajectory.length() - 1);
  ts_ = new_ts;

  MatrixXd new_ys(new_size, dim());
  new_ys << ys_, trajectory.ys().block(1, 0, trajectory.length() - 1, dim());
  ys_ = new_ys;

  MatrixXd new_yds(new_size, dim());
  new_yds << yds_, trajectory.yds().block(1, 0, trajectory.length() - 1, dim());
  yds_ = new_yds;

  MatrixXd new_ydds(new_size, dim());
  new_ydds << ydds_, trajectory.ydds().block(1, 0, trajectory.length() - 1, dim());
  ydds_ = new_ydds;
  
  assert(dim_misc() == trajectory.dim_misc());
  if (dim_misc()==0)
  {
    misc_.resize(new_size,0);
  }
  else
  {
    MatrixXd new_misc(new_size, dim_misc());
    new_misc << misc_, trajectory.misc().block(1, 0, trajectory.length() - 1, dim_misc());
    misc_ = new_misc;
  }
}

VectorXd Trajectory::getRangePerDim(void) const
{
  return ys_.colwise().maxCoeff().array()-ys_.colwise().minCoeff().array();
}

Trajectory Trajectory::generateMinJerkTrajectory(const VectorXd& ts, const VectorXd& y_from, const VectorXd& y_to)
{
  int n_time_steps = ts.size();
  int n_dims = y_from.size();
  
  MatrixXd ys(n_time_steps,n_dims), yds(n_time_steps,n_dims), ydds(n_time_steps,n_dims);
  
  double D =  ts[n_time_steps-1];
  ArrayXd tss = (ts/D).array(); 
    

  ArrayXd A = y_to.array()-y_from.array();
  
  for (int i_dim=0; i_dim<n_dims; i_dim++)
  {
    
    // http://noisyaccumulation.blogspot.fr/2012/02/how-to-decompose-2d-trajectory-data.html
    
    ys.col(i_dim)   = y_from[i_dim] + A[i_dim]*(  6*tss.pow(5)  -15*tss.pow(4) +10*tss.pow(3));
    
    yds.col(i_dim)  =             (A[i_dim]/D)*( 30*tss.pow(4)  -60*tss.pow(3) +30*tss.pow(2));
    
    ydds.col(i_dim) =         (A[i_dim]/(D*D))*(120*tss.pow(3) -180*tss.pow(2) +60*tss       );
  }
  
  return Trajectory(ts,ys,yds,ydds);
 
  
}

Trajectory Trajectory::generatePolynomialTrajectory(const VectorXd& ts, const VectorXd& y_from, const VectorXd& yd_from, const VectorXd& ydd_from,
  const VectorXd& y_to, const VectorXd& yd_to, const VectorXd& ydd_to)
{
  VectorXd a0 = y_from;
  VectorXd a1 = yd_from;
  VectorXd a2 = ydd_from / 2;

  VectorXd a3 = -10 * y_from - 6 * yd_from - 2.5 * ydd_from + 10 * y_to - 4 * yd_to + 0.5 * ydd_to;
  VectorXd a4 = 15 * y_from + 8 * yd_from + 2 * ydd_from - 15  * y_to  + 7 * yd_to - ydd_to;
  VectorXd a5 = -6 * y_from - 3 * yd_from - 0.5 * ydd_from  + 6 * y_to  - 3 * yd_to + 0.5 * ydd_to;

  int n_time_steps = ts.size();
  int n_dims = y_from.size();
  
  MatrixXd ys(n_time_steps,n_dims), yds(n_time_steps,n_dims), ydds(n_time_steps,n_dims);

  for (int i = 0; i < ts.size(); i++)
  {
    double t = (ts[i] - ts[0]) / (ts[n_time_steps - 1] - ts[0]);
    ys.row(i) = a0 + a1 * t + a2 * pow(t, 2) + a3 * pow(t, 3) + a4 * pow(t, 4) + a5 * pow(t, 5);
    yds.row(i) = a1 + 2 * a2 * t + 3 * a3 * pow(t, 2) + 4 * a4 * pow(t, 3) + 5 * a5 * pow(t, 4);
    ydds.row(i) = 2 * a2 + 6 * a3 * t + 12 * a4 * pow(t, 2) + 20 * a5 * pow(t, 3);
  }

  yds /= (ts[n_time_steps - 1] - ts[0]);
  ydds /= pow(ts[n_time_steps - 1] - ts[0], 2);

  return Trajectory(ts, ys, yds, ydds);
}

Trajectory Trajectory::generatePolynomialTrajectoryThroughViapoint(const VectorXd& ts, const VectorXd& y_from, const VectorXd& y_yd_ydd_viapoint, double viapoint_time, const VectorXd& y_to)
{
  
  int n_dims = y_from.size();
  assert(n_dims==y_to.size());
  assert(3*n_dims==y_yd_ydd_viapoint.size()); // Contains y, yd and ydd, so *3
  
  int n_time_steps = ts.size();
  
  int viapoint_time_step = 0;
  while (viapoint_time_step<n_time_steps && ts[viapoint_time_step]<viapoint_time)
    viapoint_time_step++;
  
  if (viapoint_time_step>=n_time_steps)
  {
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "ERROR: the time vector does not contain any time smaller than " << viapoint_time << ". Returning min-jerk trajectory WITHOUT viapoint." <<  endl;
    return Trajectory();
  }

  VectorXd yd_from        = VectorXd::Zero(n_dims);
  VectorXd ydd_from       = VectorXd::Zero(n_dims);

  VectorXd y_viapoint     = y_yd_ydd_viapoint.segment(0*n_dims,n_dims);
  VectorXd yd_viapoint    = y_yd_ydd_viapoint.segment(1*n_dims,n_dims);
  VectorXd ydd_viapoint   = y_yd_ydd_viapoint.segment(2*n_dims,n_dims);

  VectorXd yd_to          = VectorXd::Zero(n_dims);
  VectorXd ydd_to         = VectorXd::Zero(n_dims);

  Trajectory traj = Trajectory::generatePolynomialTrajectory(ts.segment(0, viapoint_time_step + 1), y_from, yd_from, ydd_from, y_viapoint, yd_viapoint, ydd_viapoint);
  traj.append(Trajectory::generatePolynomialTrajectory(ts.segment(viapoint_time_step, n_time_steps - viapoint_time_step), y_viapoint, yd_viapoint, ydd_viapoint, y_to, yd_to, ydd_to));

  return traj;
}


ostream& operator<<(std::ostream& output, const Trajectory& trajectory) {
  MatrixXd traj_matrix(trajectory.length(),1+3*trajectory.dim()+trajectory.dim_misc());
  traj_matrix << trajectory.ts_, trajectory.ys_, trajectory.yds_, trajectory.ydds_, trajectory.misc_; 
  output << traj_matrix << endl;
  return output;
}

bool Trajectory::saveToFile(string directory, string filename, bool overwrite) const
{
  MatrixXd traj_matrix(length(),1+3*dim()+dim_misc());
  traj_matrix << ts_, ys_, yds_, ydds_, misc_; 
  return saveMatrix(directory, filename, traj_matrix, overwrite);  
}

Trajectory Trajectory::readFromFile(string filename, int n_dims_misc)
{
  MatrixXd traj_matrix;
  if (!loadMatrix(filename,traj_matrix))
  { 
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "Cannot open filename '"<< filename <<"'." << endl;
    return Trajectory();
  }
  
  int n_dims       = (traj_matrix.cols()-1-n_dims_misc)/3;
  int n_time_steps = traj_matrix.rows();
  VectorXd ts; 
  MatrixXd ys, yds, ydds, misc;
  ts   = traj_matrix.block(0,0+0*n_dims,n_time_steps,1);
  ys   = traj_matrix.block(0,1+0*n_dims,n_time_steps,n_dims);
  yds  = traj_matrix.block(0,1+1*n_dims,n_time_steps,n_dims);
  ydds = traj_matrix.block(0,1+2*n_dims,n_time_steps,n_dims);
  misc = traj_matrix.block(0,1+3*n_dims,n_time_steps,n_dims_misc);
  
  return Trajectory(ts,ys,yds,ydds,misc);

}

}
