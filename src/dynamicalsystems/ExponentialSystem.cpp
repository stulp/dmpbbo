/**
 * @file   ExponentialSystem.cpp
 * @brief  ExponentialSystem class source file.
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

#include "dynamicalsystems/ExponentialSystem.hpp"

#include <cmath>
#include <vector>
#include <iostream>
#include <sstream>
#include <eigen3/Eigen/Core>

#include "dmpbbo_io/BoostSerializationToString.hpp"

#include "eigen/eigen_json.hpp"

#include <nlohmann/json.hpp>

using namespace std;
using namespace Eigen;  

namespace DmpBbo {
    
ExponentialSystem::ExponentialSystem(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr, double alpha, std::string name)
  : DynamicalSystem(1, tau, y_init, y_attr, name),
  alpha_(alpha)
{
  attractor_state_prealloc_ = VectorXd::Zero(dim_orig());
}

ExponentialSystem::~ExponentialSystem(void)
{
}

DynamicalSystem* ExponentialSystem::clone(void) const
{
  return new ExponentialSystem(tau(),initial_state(),attractor_state(),alpha_,name());
}


void ExponentialSystem::differentialEquation(
  const Eigen::Ref<const Eigen::VectorXd>& x, 
  Eigen::Ref<Eigen::VectorXd> xd) const
{
  ENTERING_REAL_TIME_CRITICAL_CODE
  // xd = alpha_*(attractor_state()-x)/tau(); // Non-realtime version now commented out  
  
  attractor_state(attractor_state_prealloc_);
  xd.noalias() = alpha_*(attractor_state_prealloc_-x)/tau();
  
  EXITING_REAL_TIME_CRITICAL_CODE
}

void ExponentialSystem::analyticalSolution(const VectorXd& ts, MatrixXd& xs, MatrixXd& xds) const
{
  int T = ts.size();
  assert(T>0);

  // Usually, we expect xs and xds to be of size T X dim(), so we resize to that. However, if the
  // input matrices were of size dim() X T, we return the matrices of that size by doing a 
  // transposeInPlace at the end. That way, the user can also request dim() X T sized matrices.
  bool caller_expects_transposed = (xs.rows()==dim() && xs.cols()==T);

  // Prepare output arguments to be of right size (Eigen does nothing if already the right size)
  xs.resize(T,dim());
  xds.resize(T,dim());
  
  VectorXd val_range = initial_state() - attractor_state();
  
  VectorXd exp_term  = -alpha_*ts/tau();
  exp_term = exp_term.array().exp().transpose();
  VectorXd pos_scale =                   exp_term;
  VectorXd vel_scale = -(alpha_/tau()) * exp_term;
  
  xs = val_range.transpose().replicate(T,1).array() * pos_scale.replicate(1,dim()).array();
  xs += attractor_state().transpose().replicate(T,1);
  xds = val_range.transpose().replicate(T,1).array() * vel_scale.replicate(1,dim()).array();
  
  if (caller_expects_transposed)
  {
    xs.transposeInPlace();
    xds.transposeInPlace();
  }
}

ExponentialSystem* ExponentialSystem::from_jsonpickle(const nlohmann::json& json) {

  double tau = from_json_to_double(json.at("tau_"));
  string name = json.at("name_");
  double alpha = from_json_to_double(json.at("alpha_"));
  
  VectorXd y_init;
  VectorXd y_attr;
  from_json(json.at("initial_state_").at("values"),y_init);
  from_json(json.at("attractor_state_").at("values"),y_attr);
  
  return new ExponentialSystem(tau,y_init,y_attr,alpha,name);
}

void to_json(nlohmann::json& j, const ExponentialSystem& obj) {
  obj.DynamicalSystem::to_json_base(j);
  
  j["alpha_"] = obj.alpha_;
  
  // for jsonpickle
  j["py/object"] = "dynamicalsystems.ExponentialSystem.ExponentialSystem";
}

string ExponentialSystem::toString(void) const
{
  nlohmann::json j;
  to_json(j,*this);
  return j.dump(4);
}

}
