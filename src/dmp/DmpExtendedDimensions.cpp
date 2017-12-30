/**
 * @file DmpExtendedDimensions.cpp
 * @brief  DmpExtendedDimensions class source file.
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
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include "dmp/DmpExtendedDimensions.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::DmpExtendedDimensions);

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <eigen3/Eigen/Core>

#include "dmp/Trajectory.hpp"
#include "functionapproximators/FunctionApproximator.hpp"
#include "dynamicalsystems/SpringDamperSystem.hpp"
#include "dynamicalsystems/ExponentialSystem.hpp"
#include "dynamicalsystems/TimeSystem.hpp"
#include "dynamicalsystems/SigmoidSystem.hpp"

#include "dmpbbo_io/EigenBoostSerialization.hpp"
#include "dmpbbo_io/BoostSerializationToString.hpp"


using namespace std;
using namespace Eigen;
using namespace Dmp;

namespace DmpBbo {

DmpExtendedDimensions::DmpExtendedDimensions(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr, 
         std::vector<FunctionApproximator*> function_approximators,
         double alpha_spring_damper, 
         DynamicalSystem* goal_system,
         DynamicalSystem* phase_system, 
         DynamicalSystem* gating_system,     
         ForcingTermScaling scaling)
  : DynamicalSystem(1, tau, y_init, y_attr, "name"),
  goal_system_(goal_system),
  phase_system_(phase_system), gating_system_(gating_system), 
  forcing_term_scaling_(scaling)
{
  initSubSystems(alpha_spring_damper, goal_system, phase_system, gating_system);
  initFunctionApproximators(function_approximators);
}

  
DmpExtendedDimensions::DmpExtendedDimensions(int n_dims_dmp, std::vector<FunctionApproximator*> function_approximators, 
   double alpha_spring_damper, DynamicalSystem* goal_system,
   DynamicalSystem* phase_system, DynamicalSystem* gating_system,     
   ForcingTermScaling scaling)
  : DynamicalSystem(1, 1.0, VectorXd::Zero(n_dims_dmp), VectorXd::Ones(n_dims_dmp), "name"),
  goal_system_(goal_system),
  phase_system_(phase_system), gating_system_(gating_system), function_approximators_(function_approximators),
  forcing_term_scaling_(scaling)
{
  initSubSystems(alpha_spring_damper, goal_system, phase_system, gating_system);
  initFunctionApproximators(function_approximators);
}
    
DmpExtendedDimensions::DmpExtendedDimensions(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr, 
         std::vector<FunctionApproximator*> function_approximators, 
         DmpType dmp_type,     
         ForcingTermScaling scaling)
  : DynamicalSystem(1, tau, y_init, y_attr, "name"),
    forcing_term_scaling_(scaling)
{  
  initSubSystems(dmp_type);
  initFunctionApproximators(function_approximators);
}
  
DmpExtendedDimensions::DmpExtendedDimensions(int n_dims_dmp, 
         std::vector<FunctionApproximator*> function_approximators, 
         DmpType dmp_type, ForcingTermScaling scaling)
  : DynamicalSystem(1, 1.0, VectorXd::Zero(n_dims_dmp), VectorXd::Ones(n_dims_dmp), "name"),
    forcing_term_scaling_(scaling)
{
  initSubSystems(dmp_type);
  initFunctionApproximators(function_approximators);
}

DmpExtendedDimensions::DmpExtendedDimensions(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr, double alpha_spring_damper, DynamicalSystem* goal_system) 
  : DynamicalSystem(1, tau, y_init, y_attr, "name"), forcing_term_scaling_(NO_SCALING)
{
  
  VectorXd one_1 = VectorXd::Ones(1);
  VectorXd one_0 = VectorXd::Zero(1);
  DynamicalSystem* phase_system  = new ExponentialSystem(tau,one_1,one_0,4);
  DynamicalSystem* gating_system = new ExponentialSystem(tau,one_1,one_0,4); 
  initSubSystems(alpha_spring_damper, goal_system, phase_system, gating_system);

  vector<FunctionApproximator*> function_approximators(y_init.size());    
  for (int dd=0; dd<y_init.size(); dd++)
    function_approximators[dd] = NULL;
  initFunctionApproximators(function_approximators);  
}



void DmpExtendedDimensions::initFunctionApproximators(vector<FunctionApproximator*> function_approximators_ext_dim)
{
  if (function_approximators.empty())
    return;
  
  // Doesn't necessarily have to be the same
  //assert(dim_orig()==(int)function_approximators_exdim.size());
  
  function_approximators_ = vector<FunctionApproximator*>(function_approximators_ext_dim.size());
  for (unsigned int dd=0; dd<function_approximators_ext_dim.size(); dd++)
  {
    if (function_approximators_ext_dim[dd]==NULL)
      function_approximators_ext_dim_[dd] = NULL;
    else
      function_approximators_ext_dim_[dd] = function_approximators_ext_dim[dd]->clone();
  }

}

DmpExtendedDimensions::~Dmp(void)
{
  delete goal_system_;   
  delete spring_system_;
  delete phase_system_;
  delete gating_system_;
  for (unsigned int ff=0; ff<function_approximators_ext_dim_.size(); ff++)
    delete (function_approximators_ext_dim_[ff]);
}

Dmp* DmpExtendedDimensions::clone(void) const {
  vector<FunctionApproximator*> function_approximators_ext_dim;
  for (unsigned int ff=0; ff<function_approximators_ext_dim_.size(); ff++)
    function_approximators_ext_dim.push_back(function_approximators_ext_dim_[ff]->clone());
  
  return new DmpExtendedDimensions(tau(), initial_state(), attractor_state(), function_approximators,
   spring_system_->damping_coefficient(), goal_system_->clone(),
   phase_system_->clone(), gating_system_->clone(), function_approximators_ext_dim);
}


void DmpExtendedDimensions::computeFunctionApproximatorOutputExtendedDimensions(const Ref<const MatrixXd>& phase_state, MatrixXd& fa_output) const
{
  /* 
  yyy
  int T = phase_state.rows();
  fa_output.resize(T,dim_orig());
  fa_output.fill(0.0);
  
  if (T>1) {
    fa_outputs_prealloc_.resize(T,dim_orig());
  }
  
  for (int i_dim=0; i_dim<dim_orig(); i_dim++)
  {
    if (function_approximators_[i_dim]!=NULL)
    {
      if (function_approximators_[i_dim]->isTrained()) 
      {
        if (T==1)
        {
          function_approximators_ext_dim_[i_dim]->predict(phase_state,fa_outputs_one_prealloc_);
          fa_output.col(i_dim) = fa_outputs_one_prealloc_;
        }
        else
        {
          function_approximators_ext_dim_[i_dim]->predict(phase_state,fa_outputs_prealloc_);
          fa_output.col(i_dim) = fa_outputs_prealloc_;
        }
      }
    }
  }
    */
}

void DmpExtendedDimensions::differentialEquation(
  const Eigen::Ref<const Eigen::VectorXd>& x, 
  Eigen::Ref<Eigen::VectorXd> xd,
  Eigen::Ref<Eigen::VectorXd> extended_dimensions) const
{
  ENTERING_REAL_TIME_CRITICAL_CODE
  Dmp::differentialEquation(x,xd);
  
  computeFunctionApproximatorOutputExtendedDimensions(x.PHASE, extended_dimensions); 

  EXITING_REAL_TIME_CRITICAL_CODE

}


void DmpExtendedDimensions::analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs, Eigen::MatrixXd& xds, Eigen::MatrixXd& fa_extended_output) const
{
  Eigen::MatrixXd forcing_terms, fa_output;
  analyticalSolution(ts, xs, xds, forcing_terms, fa_output);
  computeFunctionApproximatorOutputExtendedDimensions(xs.PHASE, fa_extended_output); 
  // get phase from xs
  // compute output of fa_extended_output
}

void DmpExtendedDimensions::analyticalSolution(const Eigen::VectorXd& ts, Trajectory& trajectory) const;
{
  Eigen::MatrixXd xs,  xds;
  analyticalSolution(ts, xs, xds);
  statesAsTrajectory(ts, xs, xds, trajectory);

  // add output fa_extended_output as misc variables
  computeFunctionApproximatorOutputExtendedDimensions(xs.PHASE, fa_extended_output); 
}

void DmpExtendedDimensions::train(const Trajectory& trajectory)
{
  train(trajectory,"");
}

void DmpExtendedDimensions::train(const Trajectory& trajectory, std::string save_directory, bool overwrite)
{
  Dmp::train(trajectory,save_directory,overwrite);

  // Get phase from trajectory
  // Get targets from trajectory
  // Train each fa_ext_dim, see below
  // Save results

  /*
  // Some checks before training function approximators
  assert(!function_approximators_.empty());
  
  for (unsigned int dd=0; dd<function_approximators_.size(); dd++)
  {
    // This is just boring stuff to figure out if and where to store the results of training
    string save_directory_dim;
    if (!save_directory.empty())
    {
      if (function_approximators_.size()==1)
        save_directory_dim = save_directory;
      else
        save_directory_dim = save_directory + "/dim" + to_string(dd);
    }
    
    // Actual training is happening here.
    VectorXd fa_target = f_target.col(dd);
    if (function_approximators_[dd]==NULL)
    {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "WARNING: function approximator cannot be trained because it is NULL." << endl;
    }
    else
    {
      if (function_approximators_[dd]->isTrained())
        function_approximators_[dd]->reTrain(fa_input_phase,fa_target,save_directory_dim,overwrite);
      else
        function_approximators_[dd]->train(fa_input_phase,fa_target,save_directory_dim,overwrite);
    }

  }
  */
}

/*
void DmpExtendedDimensions::getSelectableParameters(set<string>& selectable_values_labels) const {
  assert(function_approximators_.size()>0);
  for (int dd=0; dd<dim_orig(); dd++)
  {
    if (function_approximators_[dd]!=NULL)
    {
      if (function_approximators_[dd]->isTrained())
      {
        set<string> cur_labels;
        function_approximators_[dd]->getSelectableParameters(cur_labels);
        selectable_values_labels.insert(cur_labels.begin(), cur_labels.end());
      }
    }
  }
  selectable_values_labels.insert("goal");
  
  //cout << "selected_values_labels=["; 
  //for (string label : selected_values_labels) 
  //  cout << label << " ";
  //cout << "]" << endl;
  
}

void DmpExtendedDimensions::setSelectedParameters(const set<string>& selected_values_labels)
{
  assert(function_approximators_.size()>0);
  for (int dd=0; dd<dim_orig(); dd++)
    if (function_approximators_[dd]!=NULL)
      if (function_approximators_[dd]->isTrained())
        function_approximators_[dd]->setSelectedParameters(selected_values_labels);

  // Call superclass for initializations
  Parameterizable::setSelectedParameters(selected_values_labels);

  VectorXi lengths_per_dimension = VectorXi::Zero(dim_orig());
  for (int dd=0; dd<dim_orig(); dd++)
  {
    if (function_approximators_[dd]!=NULL)
      if (function_approximators_[dd]->isTrained())
        lengths_per_dimension[dd] = function_approximators_[dd]->getParameterVectorSelectedSize();
    
    if (selected_values_labels.find("goal")!=selected_values_labels.end())
      lengths_per_dimension[dd]++;
  }
  
  setVectorLengthsPerDimension(lengths_per_dimension);
      
}

void DmpExtendedDimensions::getParameterVectorMask(const std::set<std::string> selected_values_labels, Eigen::VectorXi& selected_mask) const
{
  assert(function_approximators_.size()>0);
  for (int dd=0; dd<dim_orig(); dd++)
  {
    assert(function_approximators_[dd]!=NULL);
    assert(function_approximators_[dd]->isTrained());
  }

  selected_mask.resize(getParameterVectorAllSize());
  selected_mask.fill(0);
  
  const int TMP_GOAL_NUMBER = -1;
  int offset = 0;
  VectorXi cur_mask;
  for (int dd=0; dd<dim_orig(); dd++)
  {
    function_approximators_[dd]->getParameterVectorMask(selected_values_labels,cur_mask);

    // This makes sure that the indices for each function approximator are different    
    int mask_offset = selected_mask.maxCoeff(); 
    for (int ii=0; ii<cur_mask.size(); ii++)
      if (cur_mask[ii]!=0)
        cur_mask[ii] += mask_offset;
        
    selected_mask.segment(offset,cur_mask.size()) = cur_mask;
    offset += cur_mask.size();
    
    // Goal
    if (selected_values_labels.find("goal")!=selected_values_labels.end())
      selected_mask(offset) = TMP_GOAL_NUMBER;
    offset++;
    
  }
  assert(offset == getParameterVectorAllSize());
  
  // Replace TMP_GOAL_NUMBER with current max value
  int goal_number = selected_mask.maxCoeff() + 1; 
  for (int ii=0; ii<selected_mask.size(); ii++)
    if (selected_mask[ii]==TMP_GOAL_NUMBER)
      selected_mask[ii] = goal_number;
    
}

int DmpExtendedDimensions::getParameterVectorAllSize(void) const
{
  int total_size = 0;
  for (unsigned int dd=0; dd<function_approximators_.size(); dd++)
    total_size += function_approximators_[dd]->getParameterVectorAllSize();
  
  // For the goal
  total_size += dim_orig();
  return total_size;
}


void DmpExtendedDimensions::getParameterVectorAll(VectorXd& values) const
{
  values.resize(getParameterVectorAllSize());
  int offset = 0;
  VectorXd cur_values;
  VectorXd attractor = attractor_state();
  for (int dd=0; dd<dim_orig(); dd++)
  {
    function_approximators_[dd]->getParameterVectorAll(cur_values);
    values.segment(offset,cur_values.size()) = cur_values;
    offset += cur_values.size();

    values(offset) = attractor(dd);
    offset++;
  }
}

void DmpExtendedDimensions::setParameterVectorAll(const VectorXd& values)
{
  assert(values.size()==getParameterVectorAllSize());
  int offset = 0;
  VectorXd cur_values;
  VectorXd attractor(dim_orig());
  for (int dd=0; dd<dim_orig(); dd++)
  {
    int n_parameters_required = function_approximators_[dd]->getParameterVectorAllSize();
    cur_values = values.segment(offset,n_parameters_required);
    function_approximators_[dd]->setParameterVectorAll(cur_values);
    offset += n_parameters_required;
    
    attractor(dd) = values(offset);
    offset += 1;
  }
  
  // Set the goal
  set_attractor_state(attractor); 
}
*/


template<class Archive>
void DmpExtendedDimensions::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Dmp);
  
  ar & BOOST_SERIALIZATION_NVP(function_approximators_ext_dim_);
}

}
