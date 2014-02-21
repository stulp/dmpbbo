/**
 * @file TimeSystem.hpp
 * @brief  TimeSystem class header file.
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

#ifndef _TIME_SYSTEM_H_
#define _TIME_SYSTEM_H_

#include "dynamicalsystems/DynamicalSystem.hpp"

namespace DmpBbo {

/** \brief Dynamical System modelling the evolution of a time: \f$\dot{x} = 1/\tau\f$.
 * In this system, time is modeled as a 1-D variable that increases with a constant velocity.
 * As with time, the system starts at 0. However, whereas time increases with 1s/s the system
 * increases with (1/tau)/s. Once the system goes above tau, velocities are 0.
 * This system can also model a countdown timer. In this case, the system starts at 1, decreases
 * with velocity -1/tau, and stops decreasing when the system drops below 0.
 * 
 * \ingroup DynamicalSystems
 */
class TimeSystem : public DynamicalSystem
{
public:
  
  /**
   *  Initialization constructor.
   *  \param tau              Time constant,                cf. DynamicalSystem::tau()
   *  \param count_down       Whether timer increases (false) or decreases (true)
   *  \param name             Name for the sytem,           cf. DynamicalSystem::name()
   */
  TimeSystem(double tau, bool count_down=false, std::string name="TimeSystem");
  
  /** Destructor. */
  ~TimeSystem(void);
  
  DynamicalSystem* clone(void) const;
  
  void differentialEquation(const Eigen::VectorXd& x, Eigen::Ref<Eigen::VectorXd> xd) const;

  void analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs, Eigen::MatrixXd& xds) const;
  
	std::string toString(void) const;

	/** Accessor function for count_down.
   * \return Whether timer increases (false) or decreases (true)
   */
  inline bool count_down(void) const 
  {
    return count_down_;
  }

private:
  bool count_down_;
  
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. See \ref sec_boost_serialization_ugliness
   */
  TimeSystem(void) {};
  
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
BOOST_CLASS_EXPORT_KEY2(DmpBbo::TimeSystem, "TimeSystem");

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::TimeSystem,boost::serialization::object_serializable);

#endif // _Time_SYSTEM_H_

