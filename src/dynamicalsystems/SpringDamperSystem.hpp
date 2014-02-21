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

#include "dynamicalsystems/DynamicalSystem.hpp"

namespace DmpBbo {

/** Value indicating that the spring constant should be set such that the
 *  spring damper system is critically damped.
 */
double const CRITICALLY_DAMPED = -1.0;

/** \brief Dynamical System modelling the evolution of a spring-damper system: \f$ m\ddot{x} = -k(x-x^g) -c\dot{x}\f$.
 * 
 * http://en.wikipedia.org/wiki/Damped_spring-mass_system
 *
 * \ingroup DynamicalSystems
 */
class SpringDamperSystem : public DynamicalSystem
{
public:
  
  /**
   *  Initialization constructor.
   *  \param tau     Time constant,            cf. DynamicalSystem::tau()
   *  \param y_init  Initial state,            cf. DynamicalSystem::initial_state()
   *  \param y_attr  Attractor state,          cf. DynamicalSystem::attractor_state()
   *  \param spring_constant  Spring constant, cf. SpringDamperSystem::spring_constant()
   *  \param damping_coefficient  Damping coefficient, cf. SpringDamperSystem::damping_coefficient()
   *  \param mass    Mass,                     cf. SpringDamperSystem::mass()
   *  \param name    Name for the sytem,       cf. DynamicalSystem::name()
   */
  SpringDamperSystem(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr, 
    double damping_coefficient, double spring_constant=CRITICALLY_DAMPED, double mass=1.0, std::string name="SpringDamperSystem");
    
  /** Destructor. */
  ~SpringDamperSystem(void);

  DynamicalSystem* clone(void) const;

  void differentialEquation(const Eigen::VectorXd& x, Eigen::Ref<Eigen::VectorXd> xd) const;

  void analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs, Eigen::MatrixXd& xds) const;
  
	std::string toString(void) const;
  
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
  inline void set_damping_coefficient(double damping_coefficient) {
    damping_coefficient_=damping_coefficient; 
  }
  
  /** 
   * Accessor function for spring constant.
   * \param[in] spring_constant Spring constant
   */
  inline void set_spring_constant(double spring_constant) {
    spring_constant_ = spring_constant; 
  }
  
  /** 
   * Accessor function for mass.
   * \param[in] mass Mass
   */
  inline void set_mass(double mass) { 
    mass_=mass; 
  }

private:
  /** Damping coefficient 'c' */
  double damping_coefficient_;

  /** Spring constant 'k' */
  double spring_constant_;

  /** Mass 'm' */
  double mass_;
  

private:
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. See \ref sec_boost_serialization_ugliness
   */
  SpringDamperSystem(void) {};
  
  /** Give boost serialization access to private members. */  
  friend class boost::serialization::access;
  
  /** Serialize class data members to boost archive. 
   * \param[in] ar Boost archive
   * \param[in] version Version of the class
   * See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/tutorial.html#simplecase
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    // serialize base class information
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(DynamicalSystem);
    
    ar & BOOST_SERIALIZATION_NVP(damping_coefficient_);
    ar & BOOST_SERIALIZATION_NVP(spring_constant_);
    ar & BOOST_SERIALIZATION_NVP(mass_);
  }
};

}

#include <boost/serialization/export.hpp>

/** Register this derived class. */
BOOST_CLASS_EXPORT_KEY2(DmpBbo::SpringDamperSystem, "SpringDamperSystem")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::SpringDamperSystem,boost::serialization::object_serializable)

#endif // _SPRING_DAMPER_SYSTEM_H_

