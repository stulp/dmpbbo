/**
 * @file DmpWithGainSchedules.hpp
 * @brief  DmpWithGainSchedules class header file.
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

#ifndef _DMP_WITH_GAIN_SCHEDULES_H_
#define _DMP_WITH_GAIN_SCHEDULES_H_

// This must be included before any Eigen header files are included
#include "dmp/Dmp.hpp"
#include "eigenutils/eigen_realtime_check.hpp"

/*
//Implement: extra states with an attractor.
//
//Option 1: Part of DMP.
//  Advantage: better integration
//  Downside: code becomes larges, no separation of basic/advanced code.
//
//Option 2: Subclass of DMP
//  Advantage: Advanced feature separated
//  Downside: ??


Alternative: move into Dmp

Dmp::setFunctionApproximatorsExtendedDims()
Dmp::trainExtendedDims()
Dmp::analyticalSolutionExtendedDims(const Eigen::VectorXd& ts, Trajectory&
trajectory) const; Dmp::analyticalSolutionExtendedDims(const Eigen::VectorXd&
ts, Eigen::MatrixXd& xs, Eigen::MatrixXd& xds, Eigen::MatrixXd& fa_gains) const;
Dmp::trainExtendedDims(const Trajectory& trajectory);
Dmp::trainExtendedDims(const Trajectory& trajectory, std::string save_directory,
bool overwrite=false); Dmp::predictExtendedDims(const Eigen::VectorXd& phases,
Eigen::MatrixXd& output_gains)
*/

namespace DmpBbo {

// forward declaration
class FunctionApproximator;
class Trajectory;

/** \defgroup DmpWithGainSchedules Dmp with Gain Schedules
 *  \ingroup Dmps
 *
 */

/**
Implementation of DMPs which contain extra dimensions to represent variable gain
schedules, as described in \cite buchli11learning.

These dimensions can also used to represent force profiles \cite
kalakrishnan11learning, or any other variables relevant to the DMP, but which
are not part of the dynamical system.
 */
class DmpWithGainSchedules : public Dmp {
 public:
  /** Constructor.
   * \param[in] dmp The Dmp part of the DmpWithGainSchedules
   * \param[in] function_approximators_gain_schedules Function approximators
   * that will represent the gain schedules.
   */
  DmpWithGainSchedules(
      Dmp* dmp,
      std::vector<FunctionApproximator*> function_approximators_gain_schedules);

  /** Destructor. */
  ~DmpWithGainSchedules(void);

  /** Return a deep copy of this object
   * \return A deep copy of this object
   */
  DmpWithGainSchedules* clone(void) const;

  /** Start integrating the system
   *
   * \param[out] x     - The first vector of state variables
   * \param[out] xd    - The first vector of rates of change of the state
   * variables \param[out] gains - The gains of the gain schedules
   *
   * \remarks x, xd, and gains should be of size dim() X 1. This forces you to
   * pre-allocate memory, which speeds things up (and also makes Eigen's Ref
   * functionality easier to deal with).
   */
  void integrateStart(Eigen::Ref<Eigen::VectorXd> x,
                      Eigen::Ref<Eigen::VectorXd> xd,
                      Eigen::Ref<Eigen::VectorXd> gains) const;

  /**
   * Integrate the system one time step.
   *
   * \param[in]  dt         Duration of the time step
   * \param[in]  x          Current state
   * \param[out] x_updated  Updated state, dt time later.
   * \param[out] xd_updated Updated rates of change of state, dt time later.
   * \param[out] gains      The gains of the gain schedules
   *
   * \remarks x should be of size dim() X 1. This forces you to pre-allocate
   * memory, which speeds things up (and also makes Eigen's Ref functionality
   * easier to deal with).
   */
  void integrateStep(double dt, const Eigen::Ref<const Eigen::VectorXd> x,
                     Eigen::Ref<Eigen::VectorXd> x_updated,
                     Eigen::Ref<Eigen::VectorXd> xd_updated,
                     Eigen::Ref<Eigen::VectorXd> gains) const;

  /**
   * Return analytical solution of the system at certain times (and return
   * forcing terms)
   *
   * \param[in]  ts  A vector of times for which to compute the analytical
   * solutions \param[out] xs  Sequence of state vectors. T x D or D x T matrix,
   * where T is the number of times (the length of 'ts'), and D the size of the
   * state (i.e. dim()) \param[out] xds Sequence of state vectors (rates of
   * change). T x D or D x T matrix, where T is the number of times (the length
   * of 'ts'), and D the size of the state (i.e. dim()) \param[out]
   * forcing_terms The forcing terms for each dimension, for debugging purposes
   * only. \param[out] fa_output The output of the function approximators, for
   * debugging purposes only. \param[out] fa_gains The output of the function
   * approximators for the gains.
   *
   * \remarks The output xs and xds will be of size D x T \em only if the matrix
   * x you pass as an argument of size D x T. In all other cases (i.e. including
   * passing an empty matrix) the size of x will be T x D. This feature has been
   * added so that you may pass matrices of either size.
   */
  void analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs,
                          Eigen::MatrixXd& xds, Eigen::MatrixXd& forcing_terms,
                          Eigen::MatrixXd& fa_output,
                          Eigen::MatrixXd& fa_gains) const;

  /**
   * Return analytical solution of the system at certain times
   *
   * \param[in]  ts  A vector of times for which to compute the analytical
   * solutions \param[out] trajectory The computed states as a trajectory.
   */
  void analyticalSolution(const Eigen::VectorXd& ts,
                          Trajectory& trajectory) const;

  /**
   * Train a DMP with a trajectory.
   * \param[in] trajectory The trajectory with which to train the DMP.
   * \todo Document misc variables
   */
  void train(const Trajectory& trajectory);

  /**
   * Train a DMP with a trajectory, and write results to file
   * \param[in] trajectory The trajectory with which to train the DMP.
   * \param[in] save_directory The directory to which to save the results.
   * \param[in] overwrite Overwrite existing files in the directory above
   * (default: false) \todo Document misc variables
   */
  void train(const Trajectory& trajectory, std::string save_directory,
             bool overwrite = false);

  /**
   * Return the dimensionality of the vector with gains.
   * \return The dimensionality of the vector with gains
   */
  int dim_gains(void) const { return function_approximators_gains_.size(); }

  /** \todo DmpWithGainSchedules does not yet override Parameterizable
   * interface. Thus, the functionapproximators for the extra dimensions for
   * gains cannot yet be parameterized through this interface.
   */

  /*
  void getSelectableParameters(std::set<std::string>& selectable_values_labels)
  const; void setSelectedParameters(const std::set<std::string>&
  selected_values_labels);

  int getParameterVectorAllSize(void) const;
  void getParameterVectorAll(Eigen::VectorXd& values) const;
  void setParameterVectorAll(const Eigen::VectorXd& values);
  void getParameterVectorMask(const std::set<std::string>
  selected_values_labels, Eigen::VectorXi& selected_mask) const;
  */

  /** Compute the outputs of the function approximators.
   * \param[in] phase_state The phase states for which the outputs are computed.
   * \param[out] fa_output The outputs of the function approximators.
   */
  virtual void computeFunctionApproximatorOutputExtendedDimensions(
      const Eigen::Ref<const Eigen::MatrixXd>& phase_state,
      Eigen::MatrixXd& fa_output) const;

 protected:
  /** Get a pointer to the function approximator for a certain dimension.
   * \param[in] i_dim Dimension for which to get the function approximator
   * \return Pointer to the function approximator.
   */
  inline FunctionApproximator* function_approximator_gains(int i_dim) const
  {
    assert(i_dim < (int)function_approximators_gains_.size());
    return function_approximators_gains_[i_dim];
  }

 private:
  /** The function approximators, one for each extra dimension.
   */
  std::vector<FunctionApproximator*> function_approximators_gains_;

  void initFunctionApproximatorsExtDims(
      std::vector<FunctionApproximator*> function_approximators);

  /** Pre-allocated memory to avoid allocating it during run-time. To enable
   * real-time. */
  mutable Eigen::MatrixXd fa_gains_outputs_one_prealloc_;

  /** Pre-allocated memory to avoid allocating it during run-time. To enable
   * real-time. */
  mutable Eigen::MatrixXd fa_gains_outputs_prealloc_;

  /** Pre-allocated memory to avoid allocating it during run-time. To enable
   * real-time. */
  mutable Eigen::MatrixXd fa_gains_output_prealloc_;

 protected:
  DmpWithGainSchedules(void){};
};

}  // namespace DmpBbo

#endif  // _DMP_WITH_GAIN_SCHEDULES_H_
