/**
 * @file   CostFunction.hpp
 * @brief  CostFunction class header file.
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

#ifndef COSTFUNCTION_H
#define COSTFUNCTION_H

#include <vector>
#include <eigen3/Eigen/Core>

namespace DmpBbo {

/** Interface for cost functions, which define a cost_function.
 * For further information see the section on \ref sec_bbo_task_and_task_solver
 */
class CostFunction
{
public:
  /** The cost function which defines the cost_function.
   *
   * \param[in] sample A sample in the search space
   * \param[out] cost The cost for the sample. The first entry cost[0] is the total cost. cost[1..n_cost_components] can contain individual cost components.
   */
  virtual void evaluate(const Eigen::VectorXd& sample, Eigen::VectorXd& cost) const = 0;

  /** Get the number of individual cost components that constitute the final total cost.
   * \return The number of cost components.
   */
  virtual unsigned int getNumberOfCostComponents(void) const = 0;
  
  /** Returns a string representation of the object.
   * \return A string representation of the object.
   */
  virtual std::string toString(void) const = 0;

  
  /** Write to output stream. 
   *  \param[in] output Output stream to which to write to 
   *  \param[in] cost_function CostFunction object to write
   *  \return Output to which the object was written 
   *
   *  \remark Calls virtual function CostFunction::toString, which must be implemented by
   * subclasses: http://stackoverflow.com/questions/4571611/virtual-operator
   */ 
  friend std::ostream& operator<<(std::ostream& output, const CostFunction& cost_function) {
    output << cost_function.toString();
    return output;
  }
};

}

#endif

