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

#define EIGEN_RUNTIME_NO_MALLOC  // Enable runtime tests for allocations

#include <eigen3/Eigen/Core>
#include <nlohmann/json_fwd.hpp>

#include "dynamicalsystems/DynamicalSystem.hpp"

namespace DmpBbo {

/** \brief Dynamical System modelling the evolution of an exponential system:
 * \f$\dot{x} = -\alpha (x-x^g)\f$.
 *
 * http://en.wikipedia.org/wiki/Exponential_decay
 *
 * http://en.wikipedia.org/wiki/Exponential_growth
 *
 * \ingroup DynamicalSystems
 */
class ExponentialSystem : public DynamicalSystem {
 public:
  /**
   *  Initialization constructor.
   *  \param tau     Time constant, cf. DynamicalSystem::tau()
   *  \param x_init  Initial state, cf. DynamicalSystem::x_init()
   *  \param x_attr  Attractor state, cf. DynamicalSystem::x_attr()
   *  \param alpha   Decay constant, cf. ExponentialSystem::alpha()
   */
  ExponentialSystem(double tau, Eigen::VectorXd x_init, Eigen::VectorXd x_attr,
                    double alpha);

  /** Destructor. */
  ~ExponentialSystem(void);

  void differentialEquation(const Eigen::Ref<const Eigen::VectorXd>& x,
                            Eigen::Ref<Eigen::VectorXd> xd) const;

  void analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs,
                          Eigen::MatrixXd& xds) const;

  /** Accessor function for decay constant.
   * \return Decay constant
   */
  double alpha(void) const { return alpha_; }

  /**
   * Accessor function for the attractor state of the dynamical system.
   * \param[out] x_attr Attractor state of the dynamical system.
   */
  inline void get_x_attr(Eigen::VectorXd& x_attr) const { x_attr = x_attr_; }

  /** Mutator function for the attractor state of the dynamical system.
   *  \param[in] x_attr Attractor state of the dynamical system.
   */
  inline void set_x_attr(const Eigen::Ref<const Eigen::VectorXd>& x_attr)
  {
    assert(x_attr.size() == dim());
    x_attr_ = x_attr;
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
    assert(y_attr.size() == dim());
    y_attr_ = y_attr;
  }

  /** Read an object from json.
   *  \param[in]  j   json input
   *  \param[out] obj The object read from json
   *
   * See also: https://github.com/nlohmann/json/issues/1324
   */
  friend void from_json(const nlohmann::json& j, ExponentialSystem*& obj);

  /** Write an object to json.
   *  \param[in] obj The object to write to json
   *  \param[out]  j json output
   *
   * See also:
   *   https://github.com/nlohmann/json/issues/1324
   *   https://github.com/nlohmann/json/issues/716
   */
  inline friend void to_json(nlohmann::json& j,
                             const ExponentialSystem* const& obj)
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
  Eigen::VectorXd x_attr_;

  /** Decay constant */
  double alpha_;
};

}  // namespace DmpBbo

#endif  // _EXPONENTIAL_SYSTEM_H_
