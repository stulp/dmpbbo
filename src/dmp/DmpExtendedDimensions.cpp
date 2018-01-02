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

/** Extracts the phase variable (1-D) from a state vector, e.g. state.PHASE */ 
#define PHASE     segment(3*dim_orig()+0,       1)
/** Extracts first T (time steps) states of the phase system, e.g. states.PHASEM(100) */ 
#define PHASEM(T)     block(0,3*dim_orig()+0,T,       1)

//Implement: extra states with an attractor. 
//
//Option 1: Part of DMP. 
//  Advantage: better integration
//  Downside: code becomes larges, no separation of basic/advanced code.
//  
//Option 2: Subclass of DMP
//  Advantage: Advanced feature separated
//  Downside: ??
  
namespace DmpBbo {

//train(xs,targets,xs_extra)
//
//differentialEquation(xs,xds,xs_extra)
//  Dmp::differentialEquation(xs,xds)
//  for fa in fas
//    xs_extra.row() = fa.predict(xs[0])
//  

DmpExtendedDimensions::DmpExtendedDimensions(
  Dmp* dmp,
  std::vector<FunctionApproximator*> function_approximators_ext_dims
) : Dmp(*dmp) 
{
  initFunctionApproximatorsExtDims(function_approximators_ext_dims);
  
  // Pre-allocate memory for real-time execution
  fa_ext_dim_outputs_one_prealloc_ = MatrixXd(1,dim_orig());
  fa_ext_dim_outputs_prealloc_ = MatrixXd(1,dim_orig());
  fa_ext_dim_output_prealloc_ = MatrixXd(1,dim_orig()); 
}
  
    
    
/*
DmpExtendedDimensions::DmpExtendedDimensions(
  double tau, 
  Eigen::VectorXd y_init, 
  Eigen::VectorXd y_attr, 
  std::vector<FunctionApproximator*> function_approximators,
  double alpha_spring_damper, 
  DynamicalSystem* goal_system,
  DynamicalSystem* phase_system, 
  DynamicalSystem* gating_system,     
  std::vector<FunctionApproximator*> function_approximators_ext_dims,
  ForcingTermScaling scaling)
  : Dmp( tau, y_init, y_attr,function_approximators,alpha_spring_damper,goal_system,phase_system,gating_system,scaling) 
{
  initFunctionApproximatorsExtDims(function_approximators_ext_dims);
}

  
DmpExtendedDimensions::DmpExtendedDimensions(int n_dims_dmp, std::vector<FunctionApproximator*> function_approximators, 
   double alpha_spring_damper, DynamicalSystem* goal_system,
   DynamicalSystem* phase_system, DynamicalSystem* gating_system,     
   std::vector<FunctionApproximator*> function_approximators_ext_dims,
   ForcingTermScaling scaling)
  : Dmp(n_dims_dmp,function_approximators,alpha_spring_damper,goal_system,phase_system,gating_system,scaling) 
{
  initFunctionApproximatorsExtDims(function_approximators_ext_dims);
}
    
DmpExtendedDimensions::DmpExtendedDimensions(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr, 
         std::vector<FunctionApproximator*> function_approximators, 
         DmpType dmp_type,     
         std::vector<FunctionApproximator*> function_approximators_ext_dims,
         ForcingTermScaling scaling)
  : Dmp(tau, y_init, y_attr,function_approximators,dmp_type,scaling),
    forcing_term_scaling_(scaling)
{  
  initFunctionApproximatorsExtDims(function_approximators_ext_dims);
}
  
DmpExtendedDimensions::DmpExtendedDimensions(int n_dims_dmp, 
         std::vector<FunctionApproximator*> function_approximators, 
         DmpType dmp_type, ForcingTermScaling scaling)
  : Dmp(n_dims_dmp, function_approximators, dmp_type,scaling)
{
  initFunctionApproximatorsExtDims(function_approximators_ext_dims);
}

*/

void DmpExtendedDimensions::initFunctionApproximatorsExtDims(vector<FunctionApproximator*> function_approximators_ext_dims)
{
  if (function_approximators_ext_dims.empty())
    return;
  
  // Doesn't necessarily have to be the same
  //assert(dim_orig()==(int)function_approximators_exdim.size());
  
  function_approximators_ext_dims_ = vector<FunctionApproximator*>(function_approximators_ext_dims.size());
  for (unsigned int dd=0; dd<function_approximators_ext_dims.size(); dd++)
  {
    if (function_approximators_ext_dims[dd]==NULL)
      function_approximators_ext_dims_[dd] = NULL;
    else
      function_approximators_ext_dims_[dd] = function_approximators_ext_dims[dd]->clone();
  }

}

DmpExtendedDimensions::~DmpExtendedDimensions(void)
{
  for (unsigned int ff=0; ff<function_approximators_ext_dims_.size(); ff++)
    delete (function_approximators_ext_dims_[ff]);
}

DmpExtendedDimensions* DmpExtendedDimensions::clone(void) const {
  
  vector<FunctionApproximator*> function_approximators_ext_dims;
  for (unsigned int ff=0; ff<function_approximators_ext_dims_.size(); ff++)
    function_approximators_ext_dims.push_back(function_approximators_ext_dims_[ff]->clone());
  
  return new DmpExtendedDimensions(Dmp::clone(), function_approximators_ext_dims);
}


void DmpExtendedDimensions::computeFunctionApproximatorOutputExtendedDimensions(const Ref<const MatrixXd>& phase_state, MatrixXd& fa_output) const
{
  int T = phase_state.rows();
  fa_output.resize(T,dim_orig());
  fa_output.fill(0.0);
  
  if (T>1) {
    fa_ext_dim_outputs_prealloc_.resize(T,dim_orig());
  }
  
  for (int i_dim=0; i_dim<dim_orig(); i_dim++)
  {
    if (function_approximators_ext_dims_[i_dim]!=NULL)
    {
      if (function_approximators_ext_dims_[i_dim]->isTrained()) 
      {
        if (T==1)
        {
          function_approximators_ext_dims_[i_dim]->predict(phase_state,fa_ext_dim_outputs_one_prealloc_);
          fa_output.col(i_dim) = fa_ext_dim_outputs_one_prealloc_;
        }
        else
        {
          function_approximators_ext_dims_[i_dim]->predict(phase_state,fa_ext_dim_outputs_prealloc_);
          fa_output.col(i_dim) = fa_ext_dim_outputs_prealloc_;
        }
      }
    }
  }
}

void DmpExtendedDimensions::integrateStep(double dt, const Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> x_updated, Eigen::Ref<Eigen::VectorXd> xd_updated, Eigen::Ref<Eigen::VectorXd> extended_dimensions) const
{
  ENTERING_REAL_TIME_CRITICAL_CODE
  Dmp::integrateStep(dt,x,x_updated,xd_updated);
  
  MatrixXd phase = x_updated.PHASE;
  MatrixXd extended_dimensions_prealloc(phase.rows(),function_approximators_ext_dims_.size());
  computeFunctionApproximatorOutputExtendedDimensions(phase, extended_dimensions_prealloc); 
  extended_dimensions = extended_dimensions_prealloc;
  
  EXITING_REAL_TIME_CRITICAL_CODE

}


void DmpExtendedDimensions::analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs, Eigen::MatrixXd& xds, Eigen::MatrixXd& fa_extended_output) const
{
  Eigen::MatrixXd forcing_terms, fa_output;
  analyticalSolution(ts, xs, xds, forcing_terms, fa_output);
  computeFunctionApproximatorOutputExtendedDimensions(xs.PHASEM(xs.rows()), fa_extended_output); 
}

void DmpExtendedDimensions::analyticalSolution(const Eigen::VectorXd& ts, Trajectory& trajectory) const
{
  Eigen::MatrixXd xs,  xds;
  Dmp::analyticalSolution(ts, xs, xds);
  statesAsTrajectory(ts, xs, xds, trajectory);

  // add output fa_extended_output as misc variables
  MatrixXd fa_extended_output;
  computeFunctionApproximatorOutputExtendedDimensions(xs.PHASEM(xs.rows()), fa_extended_output); 
  trajectory.set_misc(fa_extended_output);
}

void DmpExtendedDimensions::train(const Trajectory& trajectory)
{
  train(trajectory,"");
}

void DmpExtendedDimensions::train(const Trajectory& trajectory, std::string save_directory, bool overwrite)
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
  assert(targets.cols()==(int)function_approximators_ext_dims_.size());
  
  // Train each fa_ext_dim, see below
  // Some checks before training function approximators
  assert(!function_approximators_ext_dims_.empty());
  
  for (unsigned int dd=0; dd<function_approximators_ext_dims_.size(); dd++)
  {
    // This is just boring stuff to figure out if and where to store the results of training
    string save_directory_dim;
    if (!save_directory.empty())
    {
      if (function_approximators_ext_dims_.size()==1)
        save_directory_dim = save_directory;
      else
        save_directory_dim = save_directory + "/ext_dim" + to_string(dd);
    }
    
    // Actual training is happening here.
    VectorXd cur_target = targets.col(dd);
    if (function_approximators_ext_dims_[dd]==NULL)
    {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "WARNING: function approximator cannot be trained because it is NULL." << endl;
    }
    else
    {
      if (function_approximators_ext_dims_[dd]->isTrained())
        function_approximators_ext_dims_[dd]->reTrain(xs_phase,cur_target,save_directory_dim,overwrite);
      else
        function_approximators_ext_dims_[dd]->train(xs_phase,cur_target,save_directory_dim,overwrite);
    }

  }
}

void DmpExtendedDimensions::getSelectableParameters(set<string>& selectable_values_labels) const {
  
  Dmp::getSelectableParameters(selectable_values_labels);
  
  std::set<std::string>::iterator it;
  for (int dd=0; dd<dim_orig(); dd++)
  {
    if (function_approximators_ext_dims_[dd]!=NULL)
    {
      if (function_approximators_ext_dims_[dd]->isTrained())
      {
        set<string> cur_labels;
        function_approximators_ext_dims_[dd]->getSelectableParameters(cur_labels);

        for (it = cur_labels.begin(); it != cur_labels.end(); it++)
        { 
          string str = *it;
          str += "_ext_dims";
          selectable_values_labels.insert(str);
        }
        
      }
    }
  }
  //cout << "selected_values_labels=["; 
  //for (string label : selected_values_labels) 
  //  cout << label << " ";
  //cout << "]" << endl;
  
}

void DmpExtendedDimensions::setSelectedParameters(const set<string>& selected_values_labels)
{
  Dmp::setSelectedParameters(selected_values_labels);
  Eigen::VectorXi lengths_per_dimension_dmp = Dmp::getVectorLengthsPerDimension();
  
  Eigen::VectorXi lengths_per_dimension_ext_dim(dim_extended());
  for (int dd=0; dd<dim_extended(); dd++)
  {
    if (function_approximators_ext_dims_[dd]!=NULL)
    {
      if (function_approximators_ext_dims_[dd]->isTrained())
      {
        function_approximators_ext_dims_[dd]->setSelectedParameters(selected_values_labels);
        lengths_per_dimension_ext_dim[dd] = function_approximators_ext_dims_[dd]->getParameterVectorSelectedSize();
      }
    }
  }
  
  VectorXi lengths_per_dimension(dim_orig()+dim_extended());
  lengths_per_dimension << lengths_per_dimension_dmp, lengths_per_dimension_ext_dim;
  setVectorLengthsPerDimension(lengths_per_dimension);
      
}

void DmpExtendedDimensions::getParameterVectorMask(const std::set<std::string> selected_values_labels, Eigen::VectorXi& selected_mask) const
{
  Dmp::getParameterVectorMask(selected_values_labels,selected_mask);
  /*
  assert(function_approximators_ext_dims_.size()>0);
  for (int dd=0; dd<dim_orig(); dd++)
  {
    assert(function_approximators_ext_dims_[dd]!=NULL);
    assert(function_approximators_ext_dims_[dd]->isTrained());
  }

  selected_mask.resize(getParameterVectorAllSize());
  
  int offset = 0;
  VectorXi cur_mask;
  for (int dd=0; dd<dim_orig(); dd++)
  {
    function_approximators_ext_dims_[dd]->getParameterVectorMask(selected_values_labels,cur_mask);

    // This makes sure that the indices for each function approximator are different    
    int mask_offset = selected_mask.maxCoeff(); 
    for (int ii=0; ii<cur_mask.size(); ii++)
      if (cur_mask[ii]!=0)
        cur_mask[ii] += mask_offset;
        
    selected_mask.segment(offset,cur_mask.size()) = cur_mask;
    offset += cur_mask.size();
    
    offset++;
    
  }
  assert(offset == getParameterVectorAllSize());
  
  // Replace TMP_GOAL_NUMBER with current max value
  int goal_number = selected_mask.maxCoeff() + 1; 
  for (int ii=0; ii<selected_mask.size(); ii++)
    if (selected_mask[ii]==TMP_GOAL_NUMBER)
      selected_mask[ii] = goal_number;
    */
}

int DmpExtendedDimensions::getParameterVectorAllSize(void) const
{
  int total_size = Dmp::getParameterVectorAllSize();
  for (unsigned int dd=0; dd<function_approximators_ext_dims_.size(); dd++)
    total_size += function_approximators_ext_dims_[dd]->getParameterVectorAllSize();
  
  return total_size;
}


void DmpExtendedDimensions::getParameterVectorAll(VectorXd& values) const
{
  Dmp::getParameterVectorAll(values);

  int offset = values.size();

  values.resize(getParameterVectorAllSize());
  
  VectorXd cur_values;
  for (int dd=0; dd<dim_extended(); dd++)
  {
    function_approximators_ext_dims_[dd]->getParameterVectorAll(cur_values);
    values.segment(offset,cur_values.size()) = cur_values;
    offset += cur_values.size();
  }
}

void DmpExtendedDimensions::setParameterVectorAll(const VectorXd& values)
{
  assert(values.size()==getParameterVectorAllSize());
  
  int last_index = values.size(); // Offset at the end
  VectorXd cur_values;
  for (int dd=dim_extended()-1; dd>=0; dd--)
  {
    int n_parameters_required = function_approximators_ext_dims_[dd]->getParameterVectorAllSize();
    cur_values = values.segment(last_index-n_parameters_required,n_parameters_required);
    function_approximators_ext_dims_[dd]->setParameterVectorAll(cur_values);
    last_index -= n_parameters_required;
  }
  
  VectorXd values_for_dmp = values.segment(0,last_index);
  Dmp::setParameterVectorAll(values);
  
}


template<class Archive>
void DmpExtendedDimensions::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Dmp);
  
  ar & BOOST_SERIALIZATION_NVP(function_approximators_ext_dims_);
}

}
