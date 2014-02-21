/**
 * @file   TaskViapoint.hpp
 * @brief  TaskViapoint class header file.
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

#ifndef TASKVIAPOINT_H
#define TASKVIAPOINT_H

#include <eigen3/Eigen/Core>

#include "dmp_bbo/TaskWithTrajectoryDemonstrator.hpp"

#include "utilities/EigenBoostSerialization.hpp"

/** If the viapoint_time is set to NO_VIAPOINT_TIME, we do not compute the distance between the trajectory and the viapoint at "viapoint_time", but use the minimum distance instead. */
#define NO_VIAPOINT_TIME -1

namespace DmpBbo {
  
/**
 * Task for passing through a viapoint with minimal acceleration.
 */
class TaskViapoint : public TaskWithTrajectoryDemonstrator
{
private:
  
  Eigen::VectorXd viapoint_;
  double   viapoint_time_;
  
  Eigen::VectorXd goal_;
  double   goal_time_;
  
  double   viapoint_weight_;
  double   acceleration_weight_;
  double   goal_weight_;
  
public:
  /** Constructor.
   * \param[in] viapoint The viapoint to which to pass through.
   * \param[in] viapoint_time The time at which to pass through the viapoint.
   */
  TaskViapoint(const Eigen::VectorXd& viapoint, double  viapoint_time);
  
  /** Constructor.
   * \param[in] viapoint The viapoint to which to pass through.
   * \param[in] viapoint_time The time at which to pass through the viapoint.
   * \param[in] goal The goal to reach at the end of the movement
   * \param[in] goal_time The time at which the goal should have been reached
   */
  TaskViapoint(const Eigen::VectorXd& viapoint, double  viapoint_time, const Eigen::VectorXd& goal, double goal_time);
  
  void evaluate(const Eigen::MatrixXd& cost_vars, const Eigen::MatrixXd& task_parameters, Eigen::VectorXd& costs) const;
  
  /** Set the relative weights of the components of the cost function.
   * \param[in] viapoint_weight Weight for the cost related to not passing through the viapoint
   * \param[in] acceleration_weight Weight for the cost of accelerations
   * \param[in] goal_weight Weight for the cost of not being at the goal at the end of the movement
   */
  void setCostFunctionWeighting(double viapoint_weight, double acceleration_weight, double goal_weight=0.0);
  
  void generateDemonstration(const Eigen::MatrixXd& task_parameters, const Eigen::VectorXd& ts, Trajectory& demonstration) const;
  
  /** Returns a string representation of the object.
   * \return A string representation of the object.
   */
	std::string toString(void) const;

  /** Save a python script that is able to visualize the rollouts, given the cost-relevant variables
   *  stored in a file.
   *  \param[in] directory Directory in which to save the python script
   *  \return true if saving the script was successful, false otherwise
   */
  bool savePerformRolloutsPlotScript(std::string directory) const;

private:
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  TaskViapoint(void) {};

  /** Give boost serialization access to private members. */  
  friend class boost::serialization::access;
  
  /** Serialize class data members to boost archive. 
   * \param[in] ar Boost archive
   * \param[in] version Version of the class
   * See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/tutorial.html#simplecase
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    // serialize base class information
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(TaskWithTrajectoryDemonstrator);
    
    ar & BOOST_SERIALIZATION_NVP(viapoint_);
    ar & BOOST_SERIALIZATION_NVP(viapoint_time_);    
    ar & BOOST_SERIALIZATION_NVP(goal_);
    ar & BOOST_SERIALIZATION_NVP(goal_time_);
    ar & BOOST_SERIALIZATION_NVP(viapoint_weight_);    
    ar & BOOST_SERIALIZATION_NVP(acceleration_weight_);
    ar & BOOST_SERIALIZATION_NVP(goal_weight_);
  }

};

}

#include <boost/serialization/export.hpp>
/** Register this derived class. */
BOOST_CLASS_EXPORT_KEY2(DmpBbo::TaskViapoint, "TaskViapoint")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::TaskViapoint,boost::serialization::object_serializable);

#endif

