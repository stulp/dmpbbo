/**
 * @file SigmoidSystem.hpp
 * @brief  SigmoidSystem class header file.
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

#ifndef _SIGMOID_SYSTEM_H_
#define _SIGMOID_SYSTEM_H_

#include "dynamicalsystems/DynamicalSystem.hpp"

#include <nlohmann/json_fwd.hpp>

namespace DmpBbo {           

/** \brief Dynamical System modelling the evolution of a sigmoidal system \f$\dot{x} = -\alpha x(1-x/K)\f$.
 *
 * \ingroup DynamicalSystems
 */
class SigmoidSystem : public DynamicalSystem
{
public:

  /**
   *  Initialization constructor for a 1D system.
   *  \param tau              Time constant,                cf. DynamicalSystem::tau()
   *  \param x_init           Initial state,                cf. DynamicalSystem::initial_state()
   *  \param max_rate         Maximum rate of change,       cf. SigmoidSystem::max_rate()
   *  \param inflection_point_time Time at which maximum rate of change is achieved,  cf. SigmoidSystem::inflection_point_time()
   *  \param name             Name for the sytem,           cf. DynamicalSystem::name()
   */
   SigmoidSystem(double tau, const Eigen::VectorXd& x_init, double max_rate, double inflection_point_time, std::string name="SigmoidSystem");
  
  /** Destructor. */
  ~SigmoidSystem(void);

  static SigmoidSystem* from_jsonpickle(const nlohmann::json& json);
  
  DynamicalSystem* clone(void) const;

   void differentialEquation(
     const Eigen::Ref<const Eigen::VectorXd>& x, 
     Eigen::Ref<Eigen::VectorXd> xd
   ) const;
 
  void analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs, Eigen::MatrixXd& xds) const;

  void set_tau(double tau);
  void set_initial_state(const Eigen::VectorXd& y_init);

	std::string toString(void) const;

private:
  static Eigen::VectorXd computeKs(const Eigen::VectorXd& N_0s, double r, double inflection_point_time_time);
  
  double max_rate_;
  double inflection_point_time_;
  Eigen::VectorXd Ks_;
  
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. See \ref sec_boost_serialization_ugliness
   */
  SigmoidSystem(void) {};
  
public:
  friend void to_json(nlohmann::json& j, const SigmoidSystem& p);
  //friend void from_json(const nlohmann::json& j, SigmoidSystem& p);

};

}

#endif // _Sigmoid_SYSTEM_H_

