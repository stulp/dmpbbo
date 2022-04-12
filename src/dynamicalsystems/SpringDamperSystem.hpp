/**
 * @file SpringDamperSystem.hpp
 * @brief  SpringDamperSystem class header file.
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

#ifndef _SPRING_DAMPER_SYSTEM_H_
#define _SPRING_DAMPER_SYSTEM_H_

#define EIGEN_RUNTIME_NO_MALLOC  // Enable runtime tests for allocations

#include <eigen3/Eigen/Core>
#include <nlohmann/json_fwd.hpp>

#include "dynamicalsystems/DynamicalSystem.hpp"

namespace DmpBbo {

/** Value indicating that the spring constant should be set such that the
 *  spring damper system is critically damped.
 */
double const CRITICALLY_DAMPED = -1.0;

/** \brief Dynamical System modelling the evolution of a spring-damper system:
 * \f$ m\ddot{x} = -k(x-x^g) -c\dot{x}\f$.
 *
 * http://en.wikipedia.org/wiki/Damped_spring-mass_system
 *
 * \ingroup DynamicalSystems
 */
class SpringDamperSystem : public DynamicalSystem {
 public:
  /**
   *  Initialization constructor.
   *  \param tau     Time constant,            cf. DynamicalSystem::tau()
   *  \param y_init  Initial state,            cf.
   * DynamicalSystem::x_init() \param y_attr  Attractor state, cf.
   * DynamicalSystem::y_attr() \param spring_constant  Spring constant,
   * cf. SpringDamperSystem::spring_constant() \param damping_coefficient
   * Damping coefficient, cf. SpringDamperSystem::damping_coefficient() \param
   * mass    Mass,                     cf. SpringDamperSystem::mass()
   */
  SpringDamperSystem(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr,
                     double damping_coefficient,
                     double spring_constant = CRITICALLY_DAMPED,
                     double mass = 1.0);

  /** Destructor. */
  ~SpringDamperSystem(void);

  void differentialEquation(const Eigen::Ref<const Eigen::VectorXd>& x,
                            Eigen::Ref<Eigen::VectorXd> xd) const;

  void analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs,
                          Eigen::MatrixXd& xds) const;

  /**
   * Accessor function for damping coefficient.
   * \return Damping coefficient
   */
  inline double damping_coefficient(void) { return damping_coefficient_; }

  /**
   * Accessor function for spring constant.
   * \return Spring constant
   */
  inline double spring_constant(void) { return spring_constant_; }

  /**
   * Accessor function for mass.
   * \return Mass
   */
  inline double mass(void) { return mass_; }

  /**
   * Accessor function for damping coefficient.
   * \param[in] damping_coefficient Damping coefficient
   */
  inline void set_damping_coefficient(double damping_coefficient)
  {
    damping_coefficient_ = damping_coefficient;
  }

  /**
   * Accessor function for spring constant.
   * \param[in] spring_constant Spring constant
   */
  inline void set_spring_constant(double spring_constant)
  {
    spring_constant_ = spring_constant;
  }

  /**
   * Accessor function for mass.
   * \param[in] mass Mass
   */
  inline void set_mass(double mass) { mass_ = mass; }

  /**
   * Accessor function for the initial state of the dynamical system.
   * \param[out] y_init Initial state of the dynamical system.
   */
  inline void get_y_init(Eigen::VectorXd& y_init) const
  {
    // x = [y z], return only y part
    y_init = x_init().segment(0, dim() / 2);
  }

  /** Mutator function for the initial state of the dynamical system.
   *  \param[in] y_init Initial state of the dynamical system.
   */
  inline void set_y_init(const Eigen::Ref<const Eigen::VectorXd>& y_init)
  {
    set_x_init(y_init);
  }

  /**
   * Accessor function for the attractor state of the dynamical system.
   * \param[out] y_attr Attractor state of the dynamical system.
   */
  inline void get_y_attr(Eigen::VectorXd& y_attr) const { y_attr = y_attr_; }

  /** Mutator function for the attractor state of the dynamical system.
   *  \param[in] y_attr Attractor state of the dynamical system.
   */
  inline void set_y_attr(const Eigen::Ref<const Eigen::VectorXd>& y_attr)
  {
    // Order 2 system: attractor state is half the size of state dimensionality
    assert(y_attr.size() == dim() / 2);
    y_attr_ = y_attr;
  }

  /** Read an object from json.
   *  \param[in]  j   json input
   *  \param[out] obj The object read from json
   *
   * See also: https://github.com/nlohmann/json/issues/1324
   */
  friend void from_json(const nlohmann::json& j, SpringDamperSystem*& obj);

  /** Write an object to json.
   *  \param[in] obj The object to write to json
   *  \param[out]  j json output
   *
   * See also:
   *   https://github.com/nlohmann/json/issues/1324
   *   https://github.com/nlohmann/json/issues/716
   */
  inline friend void to_json(nlohmann::json& j,
                             const SpringDamperSystem* const& obj)
  {
    obj->to_json_helper(j);
  }

 private:
  /** Write this object to json.
   *  \param[out]  j json output
   *
   * See also:
   *   https://github.com/nlohmann/json/issues/1324
   *   https://github.com/nlohmann/json/issues/716
   */
  void to_json_helper(nlohmann::json& j) const;

  /** The attractor state of the system, to which the system will converge. */
  Eigen::VectorXd y_attr_;

  /** Damping coefficient 'c' */
  double damping_coefficient_;

  /** Spring constant 'k' */
  double spring_constant_;

  /** Mass 'm' */
  double mass_;

  /** Pre-allocated memory to avoid allocating it during run-time. To enable
   * real-time execution of the differentialEquation() function. */
  mutable Eigen::VectorXd y_, z_, yd_, zd_;
};

}  // namespace DmpBbo

#endif  // _SPRING_DAMPER_SYSTEM_H_
