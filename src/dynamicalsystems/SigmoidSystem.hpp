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

  DynamicalSystem* clone(void) const;

  void differentialEquation(const Eigen::VectorXd& x, Eigen::Ref<Eigen::VectorXd> xd) const;
 
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
  
  /** Give boost serialization access to private members. */  
  friend class boost::serialization::access;
  
  /** Serialize class data members to boost archive. 
   * \param[in] ar Boost archive
   * \param[in] version Version of the class
   * See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/tutorial.html#simplecase
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version);

};

}

#include <boost/serialization/export.hpp>

/** Register this derived class. */
BOOST_CLASS_EXPORT_KEY2(DmpBbo::SigmoidSystem, "SigmoidSystem")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::SigmoidSystem,boost::serialization::object_serializable)


#endif // _Sigmoid_SYSTEM_H_

