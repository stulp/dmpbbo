/**
 * @file   SigmoidSystem.cpp
 * @brief  SigmoidSystem class source file.
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
#define EIGEN2_SUPPORT
#include <boost/serialization/export.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include "dynamicalsystems/SigmoidSystem.hpp"

BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::SigmoidSystem);

#include <cmath>
#include <vector>
#include <iostream>
//#include <assert.h>
#include <eigen3/Eigen/Core>

#include "dmpbbo_io/EigenBoostSerialization.hpp"
#include "dmpbbo_io/BoostSerializationToString.hpp"

using namespace std;
using namespace Eigen;

namespace DmpBbo {

SigmoidSystem::SigmoidSystem(double tau, const VectorXd& x_init, double max_rate, double inflection_point_time, string name)
: DynamicalSystem(1, tau, x_init, VectorXd::Zero(x_init.size()), name),
  max_rate_(abs(max_rate)),
  inflection_point_time_(inflection_point_time)
{
    /// Tried to add this assert, but it doesn't work??? -> assert((max_rate_<50.) && (max_rate_>0.0) && "Max_Rate should be <50 and >0. I suggest using 40!");
    K_ = x_init;

    max_rate_ /= tau; // Normalize the slope over any timescale
    SigmoidSystem::initializeSystem(tau);
}

SigmoidSystem::~SigmoidSystem(void)
{
}

DynamicalSystem* SigmoidSystem::clone(void) const
{
  return new SigmoidSystem(tau(),initial_state(),max_rate_,inflection_point_time_,name());
}

void SigmoidSystem::initializeSystem(double tau_tmp)
{
    int steps = round((tau_tmp - 0.0) / 0.001); // ensure a timeline with step size ~= 0.001
    VectorXd ts = VectorXd::LinSpaced(steps, 0.0, tau_tmp);
    MatrixXd xs(ts.rows(), dim());
    MatrixXd xds(ts.rows(), dim());
    SigmoidSystem::analyticalSolution(ts, xs, xds); // integrate analytical solution to determine differential equations starting point
    VectorXd num_y_init = xs.row(1); // pull the first value from the integration
    SigmoidSystem::set_initial_state(num_y_init); // sets the differential system != 1 but on the right slope to follow the analytical system
}

void SigmoidSystem::set_tau(double new_tau) {

  // Get previous tau from superclass with tau() and set it with set_tau()
  double prev_tau = tau();

  DynamicalSystem::set_tau(new_tau);

  inflection_point_time_ = new_tau*inflection_point_time_/prev_tau; /// This may have been comprimised!!! Need to check
  max_rate_ = new_tau*max_rate_/prev_tau; // Normalize the slope over any timescale
  SigmoidSystem::initializeSystem(new_tau);
}

void SigmoidSystem::set_initial_state(const VectorXd& y_init) {
  assert(y_init.size()==dim_orig());
  DynamicalSystem::set_initial_state(y_init);
}


void SigmoidSystem::differentialEquation(const VectorXd& x, Ref<VectorXd> xd) const
{
  /** -Max_rate * [x.*(1-x)] <- Must ensure that x is initialized*/
  for (int dd=0; dd<dim(); dd++){
    xd = -1.*(max_rate_)*x.array()*(1. - (x.array() / K_(dd,0)).array());
    }
}

void SigmoidSystem::analyticalSolution(const VectorXd& ts, MatrixXd& xs, MatrixXd& xds) const
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

  // Auxillary variables to improve legibility
  double r = max_rate_;
  double M = inflection_point_time_;
  VectorXd exp_rt = (r*(ts.array() - M).array()).array().exp();

  for (int dd=0; dd<dim(); dd++)
  {
    /**  x = 1./((1+exp(Max_Rate*(ts-M))))
       dx = -Max_Rate*[x.*(1-x)]
    */

    xs.block(0,dd,T,1)  = K_(dd,0) / (1+exp_rt.array()).array();
    xds.block(0,dd,T,1) = -r*xs.block(0,dd,T,1).array() * ((1 - (xs.block(0,dd,T,1).array() / K_(dd,0))).array());

  }

  if (caller_expects_transposed)
  {
    xs.transposeInPlace();
    xds.transposeInPlace();
  }
}

template<class Archive>
void SigmoidSystem::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(DynamicalSystem);

  ar & BOOST_SERIALIZATION_NVP(max_rate_);
  ar & BOOST_SERIALIZATION_NVP(inflection_point_time_);

}


string SigmoidSystem::toString(void) const
{
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("SigmoidSystem");
}

}

