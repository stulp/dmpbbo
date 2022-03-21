/**
 * @file DmpWithGainSchedules.cpp
 * @brief  DmpWithGainSchedules class source file.
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

#include "dmp/DmpWithGainSchedules.hpp"

#include "dmp/Trajectory.hpp"
#include "functionapproximators/FunctionApproximator.hpp"
#include "dynamicalsystems/SpringDamperSystem.hpp"
#include "dynamicalsystems/ExponentialSystem.hpp"
#include "dynamicalsystems/TimeSystem.hpp"
#include "dynamicalsystems/SigmoidSystem.hpp"

#include "dmpbbo_io/BoostSerializationToString.hpp"

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

/** Extracts the phase variable (1-D) from a state vector, e.g. state.PHASE */ 
#define PHASE     segment(3*dim_orig()+0,       1)
/** Extracts first T (time steps) states of the phase system, e.g. states.PHASEM(100) */ 
#define PHASEM(T)     block(0,3*dim_orig()+0,T,       1)

  
namespace DmpBbo {

DmpWithGainSchedules::DmpWithGainSchedules(
  Dmp* dmp,
  std::vector<FunctionApproximator*> function_approximators_gains
) : Dmp(*dmp) 
{
  initFunctionApproximatorsExtDims(function_approximators_gains);
  
  // Pre-allocate memory for real-time execution
  fa_gains_outputs_one_prealloc_ = MatrixXd(1,dim_orig());
  fa_gains_outputs_prealloc_ = MatrixXd(1,dim_orig());
  fa_gains_output_prealloc_ = MatrixXd(1,dim_orig()); 
}
  
    
    
/*
DmpWithGainSchedules::DmpWithGainSchedules(
  double tau, 
  Eigen::VectorXd y_init, 
  Eigen::VectorXd y_attr, 
  std::vector<FunctionApproximator*> function_approximators,
  double alpha_spring_damper, 
  DynamicalSystem* goal_system,
  DynamicalSystem* phase_system, 
  DynamicalSystem* gating_system,     
  std::vector<FunctionApproximator*> function_approximators_gains,
  ForcingTermScaling scaling)
  : Dmp( tau, y_init, y_attr,function_approximators,alpha_spring_damper,goal_system,phase_system,gating_system,scaling) 
{
  initFunctionApproximatorsExtDims(function_approximators_gains);
}

  
DmpWithGainSchedules::DmpWithGainSchedules(int n_dims_dmp, std::vector<FunctionApproximator*> function_approximators, 
   double alpha_spring_damper, DynamicalSystem* goal_system,
   DynamicalSystem* phase_system, DynamicalSystem* gating_system,     
   std::vector<FunctionApproximator*> function_approximators_gains,
   ForcingTermScaling scaling)
  : Dmp(n_dims_dmp,function_approximators,alpha_spring_damper,goal_system,phase_system,gating_system,scaling) 
{
  initFunctionApproximatorsExtDims(function_approximators_gains);
}
    
DmpWithGainSchedules::DmpWithGainSchedules(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr, 
         std::vector<FunctionApproximator*> function_approximators, 
         DmpType dmp_type,     
         std::vector<FunctionApproximator*> function_approximators_gains,
         ForcingTermScaling scaling)
  : Dmp(tau, y_init, y_attr,function_approximators,dmp_type,scaling),
    forcing_term_scaling_(scaling)
{  
  initFunctionApproximatorsExtDims(function_approximators_gains);
}
  
DmpWithGainSchedules::DmpWithGainSchedules(int n_dims_dmp, 
         std::vector<FunctionApproximator*> function_approximators, 
         DmpType dmp_type, ForcingTermScaling scaling)
  : Dmp(n_dims_dmp, function_approximators, dmp_type,scaling)
{
  initFunctionApproximatorsExtDims(function_approximators_gains);
}

*/

void DmpWithGainSchedules::initFunctionApproximatorsExtDims(vector<FunctionApproximator*> function_approximators_gains)
{
  if (function_approximators_gains.empty())
    return;
  
  // Doesn't necessarily have to be the same
  //assert(dim_orig()==(int)function_approximators_exdim.size());
  
  function_approximators_gains_ = vector<FunctionApproximator*>(function_approximators_gains.size());
  for (unsigned int dd=0; dd<function_approximators_gains.size(); dd++)
  {
    if (function_approximators_gains[dd]==NULL)
      function_approximators_gains_[dd] = NULL;
    else
      function_approximators_gains_[dd] = function_approximators_gains[dd]->clone();
  }

}

DmpWithGainSchedules::~DmpWithGainSchedules(void)
{
  for (unsigned int ff=0; ff<function_approximators_gains_.size(); ff++)
    delete (function_approximators_gains_[ff]);
}

DmpWithGainSchedules* DmpWithGainSchedules::clone(void) const {
  
  vector<FunctionApproximator*> function_approximators_gains;
  for (unsigned int ff=0; ff<function_approximators_gains_.size(); ff++)
    function_approximators_gains.push_back(function_approximators_gains_[ff]->clone());
  
  return new DmpWithGainSchedules(Dmp::clone(), function_approximators_gains);
}


void DmpWithGainSchedules::computeFunctionApproximatorOutputExtendedDimensions(const Ref<const MatrixXd>& phase_state, MatrixXd& fa_output) const
{
  int T = phase_state.rows();
  fa_output.resize(T,dim_orig());
  fa_output.fill(0.0);
  
  if (T>1) {
    fa_gains_outputs_prealloc_.resize(T,dim_orig());
  }
  
  for (int i_dim=0; i_dim<dim_orig(); i_dim++)
  {
    if (function_approximators_gains_[i_dim]!=NULL)
    {
      if (function_approximators_gains_[i_dim]->isTrained()) 
      {
        if (T==1)
        {
          function_approximators_gains_[i_dim]->predict(phase_state,fa_gains_outputs_one_prealloc_);
          fa_output.col(i_dim) = fa_gains_outputs_one_prealloc_;
        }
        else
        {
          function_approximators_gains_[i_dim]->predict(phase_state,fa_gains_outputs_prealloc_);
          fa_output.col(i_dim) = fa_gains_outputs_prealloc_;
        }
      }
    }
  }
}

void DmpWithGainSchedules::integrateStart(Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> xd, Eigen::Ref<Eigen::VectorXd> gains) const
{
  Dmp::integrateStart(x,xd);
  MatrixXd phase = x.PHASE;
  MatrixXd gains_prealloc(1,function_approximators_gains_.size());
  computeFunctionApproximatorOutputExtendedDimensions(phase, gains_prealloc); 
  gains = gains_prealloc.transpose();
}


void DmpWithGainSchedules::integrateStep(double dt, const Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> x_updated, Eigen::Ref<Eigen::VectorXd> xd_updated, Eigen::Ref<Eigen::VectorXd> gains) const
{
  ENTERING_REAL_TIME_CRITICAL_CODE
  Dmp::integrateStep(dt,x,x_updated,xd_updated);
  
  MatrixXd phase = x_updated.PHASE;
  MatrixXd gains_prealloc(1,function_approximators_gains_.size());
  computeFunctionApproximatorOutputExtendedDimensions(phase, gains_prealloc); 
  gains = gains_prealloc.transpose();
  
  EXITING_REAL_TIME_CRITICAL_CODE

}


void DmpWithGainSchedules::analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs, Eigen::MatrixXd& xds, Eigen::MatrixXd& forcing_terms, Eigen::MatrixXd& fa_output, Eigen::MatrixXd& fa_gains) const
{
  Dmp::analyticalSolution(ts, xs, xds, forcing_terms, fa_output);
  computeFunctionApproximatorOutputExtendedDimensions(xs.PHASEM(xs.rows()), fa_gains); 
}

void DmpWithGainSchedules::analyticalSolution(const Eigen::VectorXd& ts, Trajectory& trajectory) const
{
  Eigen::MatrixXd xs,  xds;
  Dmp::analyticalSolution(ts, xs, xds);
  statesAsTrajectory(ts, xs, xds, trajectory);

  // add output fa_gains as misc variables
  MatrixXd fa_gains;
  computeFunctionApproximatorOutputExtendedDimensions(xs.PHASEM(xs.rows()), fa_gains); 
  trajectory.set_misc(fa_gains);
}

void DmpWithGainSchedules::train(const Trajectory& trajectory)
{
  train(trajectory,"");
}

void DmpWithGainSchedules::train(const Trajectory& trajectory, std::string save_directory, bool overwrite)
{
  // First, train the DMP
  Dmp::train(trajectory,save_directory,overwrite);
  
  // Get phase from trajectory
  // Integrate analytically to get phase states
  MatrixXd xs_ana;
  MatrixXd xds_ana;
  Dmp::analyticalSolution(trajectory.ts(),xs_ana,xds_ana);
  MatrixXd xs_phase  = xs_ana.PHASEM(trajectory.length());


  // Get targets from trajectory
  MatrixXd targets = trajectory.misc();
  
  // The dimensionality of the extra variables in the trajectory must be the same as the number of
  // function approximators.
  assert(targets.cols()==(int)function_approximators_gains_.size());
  
  // Train each fa_gains, see below
  // Some checks before training function approximators
  assert(!function_approximators_gains_.empty());
  
  for (unsigned int dd=0; dd<function_approximators_gains_.size(); dd++)
  {
    // This is just boring stuff to figure out if and where to store the results of training
    string save_directory_dim;
    if (!save_directory.empty())
    {
      if (function_approximators_gains_.size()==1)
        save_directory_dim = save_directory;
      else
        save_directory_dim = save_directory + "/gains" + to_string(dd);
    }
    
    // Actual training is happening here.
    VectorXd cur_target = targets.col(dd);
    if (function_approximators_gains_[dd]==NULL)
    {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "WARNING: function approximator cannot be trained because it is NULL." << endl;
    }
    else
    {
      if (function_approximators_gains_[dd]->isTrained())
        function_approximators_gains_[dd]->reTrain(xs_phase,cur_target,save_directory_dim,overwrite);
      else
        function_approximators_gains_[dd]->train(xs_phase,cur_target,save_directory_dim,overwrite);
    }

  }
}


void DmpWithGainSchedules::getSelectableParameters(set<string>& selectable_values_labels) const {
  
  // Get the selectable parameters from the dmp
  Dmp::getSelectableParameters(selectable_values_labels);
  
  // Now add the ones related to the gains 
  // Add "_gains" to the labels of the function approximators to distinguish them
  // from the labels for the function approximators of the DMP
  std::set<std::string>::iterator it;
  for (int dd=0; dd<dim_orig(); dd++)
  {
    if (function_approximators_gains_[dd]!=NULL)
    {
      if (function_approximators_gains_[dd]->isTrained())
      {
        set<string> cur_labels;
        function_approximators_gains_[dd]->getSelectableParameters(cur_labels);

        for (it = cur_labels.begin(); it != cur_labels.end(); it++)
        { 
          string str = *it;
          str += "_gains";
          selectable_values_labels.insert(str);
        }
        
      }
    }
  }
  
  cout << "selectable_values_labels=["; 
  for (string label : selectable_values_labels) 
    cout << label << " ";
  cout << "]" << endl;
  
}

void DmpWithGainSchedules::setSelectedParameters(const set<string>& selected_values_labels)
{
  Dmp::setSelectedParameters(selected_values_labels);
  
  set<string> labels_gains;
  for (string label : selected_values_labels) {
      if (label.find("_gains") != string::npos) {
        label.substr(0,label.length()-6); // Remove '_gains' 
        labels_gains.insert(label);
      }
  }
      
  for (int dd=0; dd<dim_gains(); dd++)
    if (function_approximators_gains_[dd]!=NULL)
      if (function_approximators_gains_[dd]->isTrained())
        function_approximators_gains_[dd]->setSelectedParameters(labels_gains);
}


int DmpWithGainSchedules::getParameterVectorSize(void) const
{
  int size = Dmp::getParameterVectorSize();
  
  for (unsigned int dd=0; dd<function_approximators_gains_.size(); dd++)
    size += function_approximators_gains_[dd]->getParameterVectorSize();
  
  return size;
}

void DmpWithGainSchedules::getParameterVector(VectorXd& values, bool normalized) const
{
  Dmp::getParameterVector(values);

  int offset = values.size();

  values.conservativeResize(getParameterVectorSize());
  
  VectorXd cur_values;
  for (int dd=0; dd<dim_gains(); dd++)
  {
    function_approximators_gains_[dd]->getParameterVector(cur_values);
    values.segment(offset,cur_values.size()) = cur_values;
    offset += cur_values.size();
  }
}

void DmpWithGainSchedules::setParameterVector(const VectorXd& values, bool normalized)
{
  assert(values.size()==getParameterVectorSize());

  int n_params_for_dmp = Dmp::getParameterVectorSize();

  VectorXd values_for_dmp = values.segment(0,n_params_for_dmp);
  Dmp::setParameterVector(values_for_dmp);

  int offset = n_params_for_dmp;
  VectorXd cur_values;
  for (int dd=dim_gains()-1; dd>=0; dd--)
  {
    int n_parameters_required = function_approximators_gains_[dd]->getParameterVectorSize();
    cur_values = values.segment(offset,n_parameters_required);
    function_approximators_gains_[dd]->setParameterVector(cur_values,normalized);
    offset += n_parameters_required;
  }
  
  
}

/*
void DmpWithGainSchedules::setParameterVector(const std::vector<Eigen::VectorXd>& vector_values, bool normalized)
{
  VectorXd cur_values;
  VectorXd attractor(dim_orig());
  for (int dd=0; dd<dim_orig(); dd++)
  {
    cur_values = vector_values[dd];
    
    int n_fa_pars = function_approximators_[dd]->getParameterVectorSize();
    if (isParameterSelected("goal")) {
      assert(cur_values.size()==n_fa_pars+1);
      // ggg Goal is not normalized
      attractor(dd) = cur_values[n_fa_pars-1]; // goal is last value
      cur_values = cur_values.head(n_fa_pars);
      
    } else {
      assert(cur_values.size()==n_fa_pars);
      
    }
    
    function_approximators_[dd]->setParameterVector(cur_values,normalized);
  }
  
  if (isParameterSelected("goal")) {
    // Set the goal
    set_attractor_state(attractor); 
  }
}
*/


}
