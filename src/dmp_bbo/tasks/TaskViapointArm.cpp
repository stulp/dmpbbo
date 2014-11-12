/**
 * @file   TaskViapointArm.cpp
 * @brief  TaskViapointArm class source file.
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

#include <boost/serialization/export.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include "dmp_bbo/tasks/TaskViapointArm.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::TaskViapointArm);

#include <boost/serialization/base_object.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <eigen3/Eigen/Core>

#include "dmp_bbo/tasks/TaskViapoint.hpp"
#include "dmpbbo_io/EigenBoostSerialization.hpp"
#include "dmpbbo_io/BoostSerializationToString.hpp"

using namespace std;
using namespace Eigen;

namespace DmpBbo {

TaskViapointArm::TaskViapointArm(const Eigen::VectorXd& link_lengths, const Eigen::VectorXd& viapoint, double  viapoint_time)
: link_lengths_(link_lengths), viapoint_(viapoint), viapoint_time_(viapoint_time)
{
  assert(link_lengths.size()>0);
}

void TaskViapointArm::evaluate(const MatrixXd& cost_vars, const MatrixXd& task_parameters, VectorXd& costs) const
{
  // the cost_vars that are the input to this function are joint angles
  // these are converted to end-effector positions
  // these end-effector positions are then passed to TaskViapoint::evaluate() to compute the costs
  //   note 1:  TaskViapoint::evaluate() should not compute acceleration costs!
  
  
  int n_samples = cost_vars.rows();
  int n_dims_viapoint = viapoint_.size();
  
  int n_dims = link_lengths_.size();
  int n_cost_vars = 4*n_dims + 1; // y_1..y_D  yd_1..yd_D  ydd_1..ydd_D  t forcing_term_1..forcing_term_D
  int n_time_steps = cost_vars.cols()/n_cost_vars;
  // cost_vars  = n_samples x (n_time_steps*n_cost_vars)

  cout << "__________________________" << endl;
  cout << "  n_dims=" << n_dims << endl;
  cout << "  n_dims_viapoint=" << n_dims_viapoint << endl;
  cout << "  n_samples=" << n_samples << endl;
  cout << "  n_cost_vars=" << n_cost_vars << endl;
  cout << "  cost_vars.cols()=" << cost_vars.cols() << endl;  
  cout << "  n_time_steps=" << n_time_steps << endl;

  // 2D position of joints over time. Last joint is dummy joint to store end-effector position.
  MatrixXd x_joints(n_time_steps,n_dims+1);
  MatrixXd y_joints(n_time_steps,n_dims+1);
  
  
  // y_1..y_D  yd_1..yd_D  ydd_1..ydd_D  t=0
  // y_1..y_D  yd_1..yd_D  ydd_1..ydd_D  t=1
  //  |    |     |     |      |     |     |
  // y_1..y_D  yd_1..yd_D  ydd_1..ydd_D  t=T
  MatrixXd rollout; //(n_time_steps,n_cost_vars);
  MatrixXd my_row(1,n_time_steps*n_cost_vars);
  for (int k=0; k<n_samples; k++)
  {
    my_row = cost_vars.row(k);
    rollout = (Map<MatrixXd>(my_row.data(),n_cost_vars,n_time_steps)).transpose();   
    
    // rollout is of size   n_time_steps x n_cost_vars
    VectorXd ts = rollout.col(3 * n_dims);
    
    x_joints.fill(0);
    y_joints.fill(0);
    for (int tt=0; tt<n_time_steps; tt++)
    {
      double sum_angles = 0.0;
      for (int i_dof=0; i_dof<n_dims; i_dof++)
      {
        double angle = rollout(tt,i_dof*3);
        sum_angles = sum_angles + angle;
        x_joints(tt,i_dof+1) = x_joints(tt,i_dof) + cos(sum_angles)*link_lengths_(i_dof);
        y_joints(tt,i_dof+1) = y_joints(tt,i_dof) + sin(sum_angles)*link_lengths_(i_dof);
      }
    }
    
    
  }
  
  exit(0);

  // y_1..y_D  yd_1..yd_D  ydd_1..ydd_D  t=0
  // y_1..y_D  yd_1..yd_D  ydd_1..ydd_D  t=1
  //  |    |     |     |      |     |     |
  // y_1..y_D  yd_1..yd_D  ydd_1..ydd_D  t=T
  MatrixXd rollout; //(n_time_steps,n_cost_vars);
  MatrixXd my_row(1,n_time_steps*n_cost_vars);

/*  
  for (int k=0; k<n_samples; k++)
  {
    my_row = cost_vars.row(k);
    rollout = (Map<MatrixXd>(my_row.data(),n_cost_vars,n_time_steps)).transpose();   
    
    // rollout is of size   n_time_steps x n_cost_vars
    VectorXd ts = rollout.col(3 * n_dims);

    
    double sum_ydd = 0.0;
    if (acceleration_weight_!=0.0)
    {
      MatrixXd ydd = rollout.block(0,2*n_dims,n_time_steps,n_dims);
      // ydd = n_time_steps x n_dims
      sum_ydd = ydd.array().pow(2).sum();
    }

    double delay_cost = 0.0;
    if (goal_weight_!=0.0)
    {
      int goal_time_step = 0;
      while (goal_time_step < ts.size() && ts[goal_time_step] < goal_time_)
        goal_time_step++;

      const MatrixXd y_after_goal = rollout.block(goal_time_step, 0,
        rollout.rows() - goal_time_step, n_dims);

      delay_cost = (y_after_goal.rowwise() - goal_.transpose()).rowwise().squaredNorm().sum();
    }

    costs[k] =  
      viapoint_weight_*dist_to_viapoint + 
      acceleration_weight_*sum_ydd/n_time_steps + 
      goal_weight_*delay_cost;
  }
  */
}

void TaskViapointArm::setCostFunctionWeighting(double viapoint_weight, double acceleration_weight)
{
  viapoint_weight_ = viapoint_weight;
  acceleration_weight_ = acceleration_weight;
}

string TaskViapointArm::toString(void) const {
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("TaskViapointArm");
}

template<class Archive>
void TaskViapointArm::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Task);
  
  ar & BOOST_SERIALIZATION_NVP(link_lengths_);
  ar & BOOST_SERIALIZATION_NVP(viapoint_);
  ar & BOOST_SERIALIZATION_NVP(viapoint_time_);
  ar & BOOST_SERIALIZATION_NVP(viapoint_weight_);
  ar & BOOST_SERIALIZATION_NVP(acceleration_weight_);
}

bool TaskViapointArm::savePerformRolloutsPlotScript(string directory) const
{
  return true;
}


}
