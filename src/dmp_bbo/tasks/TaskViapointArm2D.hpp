/**
 * @file   TaskViapointArm2D.hpp
 * @brief  TaskViapointArm2D class header file.
 * @author Freek Stulp
 *
 * This file is part of DmpBbo, a set of libraries and programs for the 
 * black-box optimization of dynamical movement primitives.
 * Copyright (C) 2018 Freek Stulp
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

#ifndef TaskViapointArm2D_H
#define TaskViapointArm2D_H

#include <eigen3/Eigen/Core>

#include "dmp_bbo/tasks/TaskViapoint.hpp"

namespace DmpBbo {

class TaskViapointArm2D : public TaskViapoint
{
  
public:
  /** Constructor.
   * \param[in] n_dofs The number of degrees-of-freedom in the arm.
   * \param[in] viapoint The viapoint to which to pass through.
   * \param[in] viapoint_time The time at which to pass through the viapoint.
   * \param[in] viapoint_radius The distance to the viapoint within which this cost is 0
   */
  TaskViapointArm2D(int n_dofs, const Eigen::VectorXd& viapoint, double  viapoint_time=TIME_AT_MINIMUM_DIST, double viapoint_radius=0.0);

  virtual ~TaskViapointArm2D(void) {}
  
  virtual void evaluateRollout(const Eigen::MatrixXd& cost_vars, const Eigen::VectorXd& sample, const Eigen::VectorXd& task_parameters, Eigen::VectorXd& cost) const;
  
  /** Returns a string representation of the object.
   * \return A string representation of the object.
   */
	std::string toString(void) const;
	
private:
  /** The number of degrees-of-freedom in the arm. 
      Strictly speaking this variable is not necessary. But it is useful for some checks with assert.
   */
  int n_dofs_;
  
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  TaskViapointArm2D(void) {};
  

  /** Give boost serialization access to private members. */  
  friend class boost::serialization::access;
  
  /** Serialize class data members to boost archive. 
   * \param[in] ar Boost archive
   * \param[in] version Version of the class
   * See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/tutorial.html#simplecase
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version);
  
};                              

}

#include <boost/serialization/export.hpp>
/** Register this derived class. */
BOOST_CLASS_EXPORT_KEY2(DmpBbo::TaskViapointArm2D, "TaskViapointArm2D")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::TaskViapointArm2D,boost::serialization::object_serializable);

#endif

