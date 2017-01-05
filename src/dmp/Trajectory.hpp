/**
 * @file Trajectory.hpp
 * @brief  Trajectory class header file.
 * @author Freek Stulp, Thibaut Munzer
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

#ifndef _TRAJECTORY_H_
#define _TRAJECTORY_H_

#include <iosfwd>
#include <eigen3/Eigen/Core>

#include <boost/serialization/access.hpp>
#include "dmpbbo_io/EigenBoostSerialization.hpp"


namespace DmpBbo {

/** \brief A class for storing trajectories, i.e. positions, velocities and accelerations of
 *  variables over time.
 *
 * A trajectory is essentially a matrix of following form:
\verbatim
t_0   x^0_0 x^1_0 .. x^D_0   xd^0_0 xd^1_0 .. xd^D_0   xdd^0_0 xdd^1_0 ... xdd^D_0
t_1   x^0_1 x^1_1 .. x^D_1   xd^0_1 xd^1_1 .. xd^D_1   xdd^0_1 xdd^1_1 ... xdd^D_1
 :       :     :       :         :      :       :         :       :           :
t_T   x^0_T x^1_T .. x^D_T   xd^0_T xd^1_T .. xd^D_T   xdd^0_T xdd^1_T ... xdd^D_T
\endverbatim
For each time t_0 to t_T it contains the positions, velocities and accelerations of a D-dimensional state.
 */
class Trajectory
{
public:
  /** Default constructor.
   */
  Trajectory(void);
  
  /** Initializing constructor.
   * \param[in] ts Times
   * \param[in] ys Positions
   * \param[in] yds Velocities 
   * \param[in] ydds Acceleration
   * \param[in] misc Miscellaneous other variables which you may add if necessary
   */
   Trajectory(const Eigen::VectorXd& ts, const Eigen::MatrixXd& ys,  const Eigen::MatrixXd& yds,  const Eigen::MatrixXd& ydds, const Eigen::MatrixXd& misc=Eigen::MatrixXd(0,0));

  /** Accessor function for the times at which measurements were made.
   * \return Times at which measurements were made.
   */
  inline const Eigen::VectorXd& ts(void) const { return ts_; }
  
  /** Accessor function for the positions at different times
   * \return Positions at different times
   */
  inline const Eigen::MatrixXd& ys(void) const { return ys_; }
  
  /** Accessor function for the velocities at different times
   * \return Velocities at different times
   */   
  inline const Eigen::MatrixXd& yds(void) const { return yds_; }
  
  /** Accessor function for the accelerations at different times
   * \return Accelerations at different times
   */
  inline const Eigen::MatrixXd& ydds(void) const { return ydds_; }
  
  /** Return miscellaneous variables included in the trajectory.
   * This can be used for instance to store task paramaters for each time step alongside the
   * values of the trajectory variables at each time step.
   * \return Matrix of miscellaneous variables, of size n_time_steps X n_misc_variables
   */
  inline const Eigen::MatrixXd& misc(void) const { return misc_; }
  
  /** Set miscellaneous variables included in the trajectory.
   * This can be used for instance to set task paramaters for each time step alongside the
   * values of the trajectory variables at each time step.
   * \param[in] misc Matrix of miscellaneous variables, of size n_time_steps X n_misc_variables
   */
  void set_misc(const Eigen::MatrixXd& misc);
  
  /** Get the length of the trajectory, i.e. the number of time steps. 
   * \return The number of time steps
   */  
  inline int length(void) const { return ts_.size(); }
  
  /** Get the duration of the trajectory in seconds.
   * \return The duration of the trajecory in seconds.
   */  
  inline double duration(void) const {  return (ts_[ts_.size()-1]-ts_[0]); }
  
  /** Get the dimensionality of the trajectory. 
   * \return The dimensionality of the trajectory.
   */  
  inline int dim(void) const { return ys_.cols(); }
  
  /** Get the dimensionality of the  misc variables. 
   * \return The dimensionality of the misc variables.
   */  
  inline int dim_misc(void) const { return misc_.cols(); }
  
  /** Get the first state, i.e. at t=0, in the trajectory (only positions).
   *  \return First state in the trajectory.
   */
  inline Eigen::VectorXd initial_y(void) const { return ys_.row(0); }

  /** Get the last state, i.e. at t=T, in the trajectory (only positions).
   *  \return Last state in the trajectory.
   */
  inline Eigen::VectorXd final_y(void) const { return ys_.row(ys_.rows()-1); }
  
  /** Append another trajectory at the end. The appended trajectory should begin as current trajectory ends, in terms of time, position, velocity an acceleration.
   *
   * \param[in] trajectory Trajectory to append.
   */
  void append(const Trajectory& trajectory);
  
  /** Get the range of ys per dimension.
   *
   * \return The range of ys, one value for each dimension.
   */
  Eigen::VectorXd getRangePerDim(void) const;
  
  /** Write object to an output stream.
   *
   *  \param[in] output  Output stream to which to write to
   *  \param[in] trajectory Trajectory to write
   *  \return    Output stream
   */
  friend std::ostream& operator<<(std::ostream& output, const Trajectory& trajectory);

  /** Save a trajectory to a file
   * \param[in] directory Directory to which to save trajectory to
   * \param[in] filename Filename
   * \param[in] overwrite Whether to overwrite existing files (true=overwrite, false=give warning)
   * \return true if writing was successful, false otherwise.
   */
  bool saveToFile(std::string directory, std::string filename, bool overwrite=false) const;

  //friend std::istream& operator>>(std::istream& input, Trajectory& trajectory);
  
  /** Read a trajectory from a file.
   *
   *  \param[in] filename The name of the file from which to read.
   *  \param[in] n_dims_misc Number of miscellaneous variables. This is needed because the file contains a n_time_steps x N matrix, and we need to know which part of M represents the miscellaneous variables, and which part represents the trajectory. 
   *  \return Trajectory that was read
   *
   *  \todo Replace this with >>operator
   */
  static Trajectory readFromFile(std::string filename, int n_dims_misc=0);
  
  /** Generate a minimum-jerk trajectory from an initial to a final state.
   *
   * Velocities and accelerations are 0 at initial and final state.
   * 
   * \param[in] ts Times at which the state should be computed.
   * \param[in] initial_y Initial position of the trajectory. 
   * \param[in] final_y Final position of the trajectory. 
   * \return The minimum-jerk trajectory from initial_y to final_y
   *
   * See http://noisyaccumulation.blogspot.fr/2012/02/how-to-decompose-2d-trajectory-data.html
   */
  static Trajectory generateMinJerkTrajectory(const Eigen::VectorXd& ts, const Eigen::VectorXd& initial_y, const Eigen::VectorXd& final_y);


  /** Generate a fifth order polynomial trajectory from an initial to a final state.
   *
   * \param[in] ts Times at which the state should be computed.
   * \param[in] y_from Initial position of the trajectory. 
   * \param[in] yd_from Initial velocity of the trajectory. 
   * \param[in] ydd_from Initial acceleration of the trajectory. 
   * \param[in] y_to Final position of the trajectory. 
   * \param[in] yd_to Final velocity of the trajectory. 
   * \param[in] ydd_to Final acceleration of the trajectory. 
   * \return The trajectory
   */
  static Trajectory generatePolynomialTrajectory(const Eigen::VectorXd& ts, const Eigen::VectorXd& y_from, const Eigen::VectorXd& yd_from, const Eigen::VectorXd& ydd_from,
    const Eigen::VectorXd& y_to, const Eigen::VectorXd& yd_to, const Eigen::VectorXd& ydd_to);
    
  /** Generate a trajectory from an initial to a final state, through a viapoint.
   * 
   * \param[in] ts Times at which the state should be computed.
   * \param[in] y_from Initial position of the trajectory. 
   * \param[in] y_yd_ydd_viapoint Viapoint, should contain, pos, vel and acc. 
   * \param[in] viapoint_time Time at which the trajectory should pass through the viapoint. 
   * \param[in] y_to Final position of the trajectory. 
   * \return The trajectory from y_from to y_to, through the viapoint 
   */
  static Trajectory generatePolynomialTrajectoryThroughViapoint(const Eigen::VectorXd& ts, const Eigen::VectorXd& y_from, const Eigen::VectorXd& y_yd_ydd_viapoint, double viapoint_time, const Eigen::VectorXd& y_to);

private:
  Eigen::VectorXd ts_;
  Eigen::MatrixXd ys_;
  Eigen::MatrixXd yds_;
  Eigen::MatrixXd ydds_;
  Eigen::MatrixXd misc_;

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
    ar & BOOST_SERIALIZATION_NVP(ts_);
    ar & BOOST_SERIALIZATION_NVP(ys_);
    ar & BOOST_SERIALIZATION_NVP(yds_);
    ar & BOOST_SERIALIZATION_NVP(ydds_);
    ar & BOOST_SERIALIZATION_NVP(misc_);
  }

};

}

#include <boost/serialization/level.hpp>
/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::Trajectory,boost::serialization::object_serializable);

#endif


