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

#include <nlohmann/json_fwd.hpp>

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
   */
  TimeSystem(double tau, bool count_down=false);
  
  /** Destructor. */
  ~TimeSystem(void);
  
   void differentialEquation(
     const Eigen::Ref<const Eigen::VectorXd>& x, 
     Eigen::Ref<Eigen::VectorXd> xd
   ) const;

  void analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs, Eigen::MatrixXd& xds) const;
  
	/** Accessor function for count_down.
   * \return Whether timer increases (false) or decreases (true)
   */
  inline bool count_down(void) const 
  {
    return count_down_;
  }

	/** Read an object from json.
   *  \param[in]  j   json input 
   *  \param[out] obj The object read from json
   *
	 * See also: https://github.com/nlohmann/json/issues/1324
   */
  friend void from_json(const nlohmann::json& j, TimeSystem*& obj);
  
  
	/** Write an object to json.
   *  \param[in] obj The object to write to json
   *  \param[out]  j json output 
   *
	 * See also: 
	 *   https://github.com/nlohmann/json/issues/1324
	 *   https://github.com/nlohmann/json/issues/716
   */
  inline friend void to_json(nlohmann::json& j, const TimeSystem* const & obj) {
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
  
  bool count_down_;
  

};

}


#endif // _Time_SYSTEM_H_

