/**
 * @file ExponentialSystem.hpp
 * @brief  ExponentialSystem class header file.
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

#ifndef _EXPONENTIAL_SYSTEM_H_
#define _EXPONENTIAL_SYSTEM_H_

#include "dynamicalsystems/DynamicalSystem.hpp"

#include <nlohmann/json_fwd.hpp>

#include <iosfwd>


namespace DmpBbo {

/** \brief Dynamical System modelling the evolution of an exponential system: \f$\dot{x} = -\alpha (x-x^g)\f$.
 * 
 * http://en.wikipedia.org/wiki/Exponential_decay
 *
 * http://en.wikipedia.org/wiki/Exponential_growth 
 *
 * \ingroup DynamicalSystems
 */
class ExponentialSystem : public DynamicalSystem
{
public:

  /**
   *  Initialization constructor.
   *  \param tau     Time constant, cf. DynamicalSystem::tau()
   *  \param y_init  Initial state, cf. DynamicalSystem::initial_state()
   *  \param y_attr  Attractor state, cf. DynamicalSystem::attractor_state()
   *  \param alpha   Decay constant, cf. ExponentialSystem::alpha()
   */
  ExponentialSystem(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr, double alpha);
  
  static ExponentialSystem* from_jsonpickle(const nlohmann::json& json);

    
  /** Destructor. */
  ~ExponentialSystem(void);
  
   void differentialEquation(
     const Eigen::Ref<const Eigen::VectorXd>& x, 
     Eigen::Ref<Eigen::VectorXd> xd
   ) const;
  
  void analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs, Eigen::MatrixXd& xds) const;
  
  /** Accessor function for decay constant.
   * \return Decay constant
   */
  double alpha(void) const {
    return alpha_;
  }
  
	std::string toString(void) const;

private:
  
  /** Decay constant */
  double alpha_;
  
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  ExponentialSystem(void) {};

  /** Preallocated memory to make ExponentialSystem::differentialEquation() realtime. */
  mutable Eigen::VectorXd attractor_state_prealloc_;
  

public:
  friend void to_json(nlohmann::json& j, const ExponentialSystem& p);
  //friend void from_json(const nlohmann::json& j, ExponentialSystem& p);
};

}

#endif // _EXPONENTIAL_SYSTEM_H_

