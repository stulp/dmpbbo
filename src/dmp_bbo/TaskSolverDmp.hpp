/**
 * @file   TaskSolverDmp.hpp
 * @brief  TaskSolverDmp class header file.
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
 
#ifndef TaskSolverDmp_H
#define TaskSolverDmp_H

#include <string>
#include <set>
#include <eigen3/Eigen/Core>

#include "dmp_bbo/TaskSolver.hpp"

#include "dmpbbo_io/EigenBoostSerialization.hpp"

namespace DmpBbo {
  
// Forward definition
class Dmp;

/** Task solver for the viapoint task, that generates trajectories with a DMP. 
 */
class TaskSolverDmp : public TaskSolver
{
private:
  Dmp* dmp_;
  int n_time_steps_;
  double integrate_time_;
  bool use_normalized_parameter_;
  
public:
  /** Constructor.
   * \param[in] dmp The Dmp to integrate for generating trajectories (that should go through viapoints)
   * \param[in] optimize_parameters The model parameters to change in the Dmp,  cf. sec_fa_changing_modelparameters. Depends on the function approximator used for the forcing term.
   * \param[in] dt Integration time steps
   * \param[in] integrate_dmp_beyond_tau_factor If you want to integrate the Dmp for a longer duration than the tau with which it was trained, set this value larger than 1. I.e. integrate_dmp_beyond_tau_factor=1.5 will integrate for 3 seconds, if the original tau of the Dmp was 2.
   * \param[in] use_normalized_parameter Use normalized parameters, cf. sec_fa_changing_modelparameters
   */
  TaskSolverDmp(Dmp* dmp, std::set<std::string> optimize_parameters, double dt=0.01, double integrate_dmp_beyond_tau_factor=1.0, bool use_normalized_parameter=false);
    
  virtual void performRollout(const Eigen::VectorXd& samples, const Eigen::VectorXd& task_parameters, Eigen::MatrixXd& cost_vars) const;
  
  //virtual void performRollout(const std::vector<Eigen::MatrixXd>& samples_parallel, const Eigen::MatrixXd& task_parameters, Eigen::MatrixXd& cost_vars) const;

  /** Returns a string representation of the object.
   * \return A string representation of the object.
   */
	virtual std::string toString(void) const;

  /** Add a perturbation to the forcing term when computing the analytical solution.
   * \param[in] perturbation_standard_deviation Standard deviation of the normal distribution from which perturbations will be sampled.
   * 
   */
  void set_perturbation(double perturbation_standard_deviation);

private:
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  TaskSolverDmp(void) {};

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
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(TaskSolver);
    
    ar & BOOST_SERIALIZATION_NVP(dmp_);
    ar & BOOST_SERIALIZATION_NVP(n_time_steps_);    
    ar & BOOST_SERIALIZATION_NVP(integrate_time_);
    ar & BOOST_SERIALIZATION_NVP(use_normalized_parameter_);
  }

};

}

#include <boost/serialization/export.hpp>
/** Register this derived class. */
BOOST_CLASS_EXPORT_KEY2(DmpBbo::TaskSolverDmp, "TaskSolverDmp")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::TaskSolverDmp,boost::serialization::object_serializable);

#endif