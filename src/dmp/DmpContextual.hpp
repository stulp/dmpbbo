/**
 * @file DmpContextual.hpp
 * @brief  Contextual Dmp class header file.
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

#ifndef _DMP_CONTEXTUAL_H_
#define _DMP_CONTEXTUAL_H_

#include "dmp/Dmp.hpp"

#include <set>

namespace DmpBbo {

class FunctionApproximator;

/** \defgroup Dmps Dynamic Movement Primitives
 */

/** 
 * \brief Implementation of Contextual Dynamical Movement Primitives.
 *
 * Contextual Dmp extends a 'standard' Dmp by adapting to task parameters.

This is how a 'standard' Dmp would be integrated
\code 
VectorXd x, xd, x_updated;
dmp->integrateStart(x,xd);
for (int t=1; t<T; t++) {
  dmp->integrateStep(dt,x,x_updated,xd); 
  x = x_updated;
}
\endcode 

A contextual Dmp is integrated as follows.
\code 
VectorXd x, xd, x_updated;
dmp->set_task_parameters(some_task_parameters);
dmp->integrateStart(x,xd);
for (int t=1; t<T; t++) {
  dmp->integrateStep(dt,x,x_updated,xd); 
  x = x_updated;
}
\endcode 

Or, if the task parameters change over time.
\code
VectorXd x, xd, x_updated;
dmp->integrateStart(x,xd);
for (int t=1; t<T; t++) {
  dmp->set_task_parameters(some_task_parameters);
  dmp->integrateStep(dt,x,x_updated,xd); 
  x = x_updated;
}
\endcode 
 * \ingroup Dmps
 */
class DmpContextual : public Dmp
{
public:
  
  /**
   *  Initialization constructor for Contextual DMPs of known dimensionality, but with unknown
   *  initial and attractor states. Initializes the DMP with default dynamical systems.
   *  \param n_dims_dmp      Dimensionality of the DMP
   *  \param function_approximators Function approximators for the forcing term
   *  \param dmp_type  The type of DMP, see Dmp::DmpType    
   */
  DmpContextual(int n_dims_dmp, std::vector<FunctionApproximator*> function_approximators, DmpType dmp_type);
  
  /** Set the current task parameters.
   * \param[in] task_parameters Current task parameters
   */
  void set_task_parameters(const Eigen::MatrixXd& task_parameters);

  /** Set a function approximator to predict the goal from the task parameters.
   * \param[in] function_approximator The function approximator. 
   */
  void set_policy_parameter_function_goal(FunctionApproximator* function_approximator);
  
  /** Set a function approximator to predict the duration from the task parameters.
   * \param[in] function_approximator The function approximator. 
   */
  void set_policy_parameter_function_duration(FunctionApproximator* function_approximator);
  
  // Overrides Dmp::computeFunctionApproximatorOutput
  virtual void computeFunctionApproximatorOutput(const Eigen::MatrixXd& phase_state, Eigen::MatrixXd& fa_output) const = 0;
  

protected:
  // TODO: Document. Trains goal and duration first, then calls training for rest. 
  void  trainLocal(const std::vector<Trajectory>& trajectories, const std::vector<Eigen::MatrixXd>& task_parameters, std::string save_directory, bool overwrite);

public:
  /** Train a contextual Dmp with a set of trajectories (and save results to file)
   * This function is useful for debugging, i.e. if you want to save intermediate results to a
   * directory
   * \param[in] trajectories The set of trajectories
   * \param[in] task_parameters The task parameters for each of the trajectories.
   * \param[in] save_directory Directory to which to save intermediate results.
   * \param[in] overwrite Overwrite existing files in the directory above
   * Overloads Dmp::train
   */
  virtual void  train(const std::vector<Trajectory>& trajectories, const std::vector<Eigen::MatrixXd>& task_parameters, std::string save_directory, bool overwrite) = 0;
  
  /** Train a contextual DMP with multiple trajectories
   * \param[in] trajectories A set of demonstrated trajectories 
   * \param[in] task_parameters The task_parameters for each trajectory. It is a std::vector, where each task parameter element corresponds to one Trajectory. Each element is a matrix of size n_time_steps x n_task_paramaters 
   * \param[in] save_directory Directory to which to save intermediate results. Does not overwrite existing files.
   */
  void train(const std::vector<Trajectory>& trajectories, const std::vector<Eigen::MatrixXd>& task_parameters, std::string save_directory);
  
  /** Train a contextual DMP with multiple trajectories
   * \param[in] trajectories A set of demonstrated trajectories 
   * \param[in] task_parameters The task_parameters for each trajectory. It is a std::vector, where each task parameter element corresponds to one Trajectory. Each element is a matrix of size n_time_steps x n_task_paramaters 
   */
  void train(const std::vector<Trajectory>& trajectories, const std::vector<Eigen::MatrixXd>& task_parameters);
  
  /** Train a contextual DMP with multiple trajectories
   * \param[in] trajectories A set of demonstrated trajectories, the task parameters are stored as miscellaneous variables in the trajectory, see also Trajectory::misc() 
   * \param[in] save_directory Directory to which to save intermediate results
   * \param[in] overwrite Overwrite existing files in the directory above
   */
  void train(const std::vector<Trajectory>& trajectories, std::string save_directory, bool overwrite);
  
  /** Train a contextual DMP with multiple trajectories
   * \param[in] trajectories A set of demonstrated trajectories, the task parameters are stored as miscellaneous variables in the trajectory, see also Trajectory::misc() 
   * \param[in] save_directory Directory to which to save intermediate results. Does not overwrite existing files.
   */
  void train(const std::vector<Trajectory>& trajectories, std::string save_directory);
  
  /** Train a contextual DMP with multiple trajectories
   * \param[in] trajectories A set of demonstrated trajectories, the task parameters are stored as miscellaneous variables in the trajectory, see also Trajectory::misc() 
   */
  void  train(const std::vector<Trajectory>& trajectories);

  

  
protected:
  /** The current task parameters.
   */
  Eigen::MatrixXd task_parameters_;

  /** Check if several trajectories have the same duration and initial/final states.
   * \param[in] trajectories A set of trajectories 
   */  
  void checkTrainTrajectories(const std::vector<Trajectory>& trajectories);
  
  /** FunctionApproximators that relate task parameters the goal of the DMP.
   */  
  std::vector<FunctionApproximator*> policy_parameter_function_goal_;
  
  /** FunctionApproximator that relates task parameters the duration of the DMP.
   */  
  FunctionApproximator* policy_parameter_function_duration_;
  

protected:
   DmpContextual(void) {};

private:
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
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Dmp);
    
    // Do not archive task_parameters_; these will change constantly, 
    // depending on the task being solved.
  }

};

}

#include <boost/serialization/export.hpp>

/** Don't add version information to archives. */
BOOST_SERIALIZATION_ASSUME_ABSTRACT(DmpBbo::DmpContextual);
 
/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::DmpContextual,boost::serialization::object_serializable);

#endif