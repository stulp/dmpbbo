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

TaskViapointArm::TaskViapointArm(const Eigen::VectorXd& link_lengths, const Eigen::VectorXd& viapoint, double  viapoint_time, double viapoint_radius)
: link_lengths_(link_lengths)
{
  assert(link_lengths.size()>0);
  task_viapoint_ = new TaskViapoint(viapoint,viapoint_time,viapoint_radius);
  // acceleration costs are for angles, not enf-effector
  task_viapoint_->acceleration_weight_ = 0.0;
}

TaskViapointArm::TaskViapointArm(const Eigen::VectorXd& link_lengths, const Eigen::VectorXd& viapoint, double  viapoint_time, const Eigen::VectorXd& goal,  double goal_time)
: link_lengths_(link_lengths)
{
  assert(link_lengths.size()>0);
  task_viapoint_ = new TaskViapoint(viapoint,viapoint_time,goal,goal_time);
  // acceleration costs are for angles, not enf-effector
  task_viapoint_->acceleration_weight_ = 0.0;
}

TaskViapointArm::~TaskViapointArm(void)
{
  delete task_viapoint_;
}

void TaskViapointArm::evaluate(const MatrixXd& cost_vars, const MatrixXd& task_parameters, VectorXd& costs) const
{
  // the cost_vars that are the input to this function are joint angles
  // these are converted to end-effector positions
  // these end-effector positions are then passed to TaskViapoint::evaluate() to compute the costs
  //   note 1:  TaskViapoint::evaluate() should not compute acceleration costs!
  
  
  int n_samples = cost_vars.rows();
  int n_dims_viapoint = task_viapoint_->viapoint_.size();
  
  int n_dims = link_lengths_.size();
  int n_cost_vars = 4*n_dims + 1; // y_1..y_D  yd_1..yd_D  ydd_1..ydd_D  t forcing_term_1..forcing_term_D
  int n_time_steps = cost_vars.cols()/n_cost_vars;
  // cost_vars  = n_samples x (n_time_steps*n_cost_vars)

  //cout << "__________________________" << endl;
  //cout << "  n_dims=" << n_dims << endl;
  //cout << "  n_dims_viapoint=" << n_dims_viapoint << endl;
  //cout << "  n_samples=" << n_samples << endl;
  //cout << "  n_cost_vars=" << n_cost_vars << endl;
  //cout << "  cost_vars.cols()=" << cost_vars.cols() << endl;  
  //cout << "  n_time_steps=" << n_time_steps << endl;

  task_viapoint_->evaluate(cost_vars, task_parameters, costs);  
  
  /*
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

    double dist_to_viapoint = 0.0;
    if (viapoint_weight_!=0.0)
    {
      if (viapoint_time_ == TIME_AT_MINIMUM_DIST)
      {
        // Don't compute the distance at some time, but rather get the minimum distance
        const MatrixXd y = rollout.block(0, 0, rollout.rows(), n_dims);
        dist_to_viapoint = (y.rowwise() - task_viapoint_->viapoint_.transpose()).rowwise().squaredNorm().minCoeff();
      }
      else
      {
        // Compute the minimum distance at a specific time
        
        // Get the time_step at which viapoint_time_step approx ts[time_step]
        int viapoint_time_step = 0;
        while (viapoint_time_step < ts.size() && ts[viapoint_time_step] < viapoint_time_)
          viapoint_time_step++;

        assert(viapoint_time_step < ts.size());

        VectorXd y_via = rollout.row(viapoint_time_step).segment(0,n_dims);
        dist_to_viapoint = sqrt((y_via-viapoint_).array().pow(2).sum());
      }
      
      if (viapoint_radius_>0.0)
      {
        // The viapoint_radius defines a radius within which the cost is always 0
        dist_to_viapoint -= viapoint_radius_;
        if (dist_to_viapoint<0.0)
          dist_to_viapoint = 0.0;
      }
    }
    
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

void TaskViapointArm::setCostFunctionWeighting(double viapoint_weight, double acceleration_weight, double goal_weight)
{
  acceleration_weight_ = acceleration_weight;
  // Why 0.0 below? task_viapoint_ should not compute acceleration costs (costs are for angles, not end-effector accelerations)
  task_viapoint_->setCostFunctionWeighting(viapoint_weight, 0.0, goal_weight);
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
  ar & BOOST_SERIALIZATION_NVP(task_viapoint_);
}


bool TaskViapointArm::savePerformRolloutsPlotScript(string directory) const
{
  return true;
}


}
