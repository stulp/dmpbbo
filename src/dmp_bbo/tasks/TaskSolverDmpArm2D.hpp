/**
 * @file   TaskSolverDmpArm2D.hpp
 * @brief  TaskSolverDmpArm2D class header file.
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
 
#ifndef TaskSolverDmpArm2D_H
#define TaskSolverDmpArm2D_H

#include <string>
#include <set>
#include <eigen3/Eigen/Core>

#include "dmp_bbo/TaskSolverDmp.hpp"

#include "dmpbbo_io/EigenBoostSerialization.hpp"
                                   
namespace DmpBbo {
  
// Forward definitions
class Dmp;

/** Task solver for the viapoint task, that generates trajectories with a DMP. 
 */
class TaskSolverDmpArm2D : public TaskSolverDmp
{
public:
  /** Constructor.
   * \param[in] dmp The Dmp to integrate for generating trajectories (that should go through viapoints)
   * \param[in] link_lengths Lengths of the links, starting at the "shoulder"
   * \param[in] optimize_parameters The model parameters to change in the Dmp,  cf. sec_fa_changing_modelparameters. Depends on the function approximator used for the forcing term.
   * \param[in] dt Integration time steps
   * \param[in] integrate_dmp_beyond_tau_factor If you want to integrate the Dmp for a longer duration than the tau with which it was trained, set this value larger than 1. I.e. integrate_dmp_beyond_tau_factor=1.5 will integrate for 3 seconds, if the original tau of the Dmp was 2.
   * \param[in] use_normalized_parameter Use normalized parameters, cf. sec_fa_changing_modelparameters
   */
  TaskSolverDmpArm2D(Dmp* dmp, const Eigen::VectorXd& link_lengths, std::set<std::string> optimize_parameters, double dt=0.01, double integrate_dmp_beyond_tau_factor=1.0, bool use_normalized_parameter=false);
    
  virtual void performRollout(const Eigen::VectorXd& sample, const Eigen::VectorXd& task_parameters, Eigen::MatrixXd& cost_vars) const;
  
  /** Get the positions of the links given the joint angles by using forward kinematics.
   * \param[in] angles Angles of the joints
   * \param[out] link_positions Get the position of each of the joints.
   */
  void anglesToLinkPositions(const Eigen::MatrixXd& angles, Eigen::MatrixXd& link_positions) const;
  
  /** Returns a string representation of the object.
   * \return A string representation of the object.
   */
	std::string toString(void) const;      
  
	/** Get the initial angles of the arm. 
	 * \param[in] n_dofs Number of degrees of freedom of the arm.
	 * \param[out] initial_angles The initial angles.
	 */
  static void getInitialAngles(unsigned int n_dofs, Eigen::VectorXd& initial_angles);
  
	/** Get the final angles of the arm, at the end of the trajectory. 
	 * \param[in] n_dofs Number of degrees of freedom of the arm.
	 * \param[out] final_angles The angles at the end of the trajectory.
	 */
  static void getFinalAngles(unsigned int n_dofs, Eigen::VectorXd& final_angles);


private:
  Eigen::VectorXd link_lengths_;
  
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  TaskSolverDmpArm2D(void) {};

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
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(TaskSolverDmp);
    
    ar & BOOST_SERIALIZATION_NVP(link_lengths_); 
  }
                                         
};

}

#include <boost/serialization/export.hpp>
/** Register this derived class. */
BOOST_CLASS_EXPORT_KEY2(DmpBbo::TaskSolverDmpArm2D, "TaskSolverDmpArm2D")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::TaskSolverDmpArm2D,boost::serialization::object_serializable);

#endif