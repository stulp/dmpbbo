/**
 * @file   Rollout.hpp
 * @brief  Rollout class header file.
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

#ifndef ROLLOUT_H
#define ROLLOUT_H   

#include <string>
#include <vector>
#include <eigen3/Eigen/Core>

#include <boost/serialization/nvp.hpp>

namespace DmpBbo {
  
/** Class for storing the information in a Rollout */
class Rollout {
  
public:
  
  /** Constructor.
   * \param[in] policy_parameters The parameters of the policy for this rollout
   */
  Rollout(const Eigen::MatrixXd& policy_parameters);
  
  /** Constructor.
   * \param[in] policy_parameters The parameters of the policy for this rollout
   * \param[in] cost_vars The cost-relevant variables that arose from executing the policy (e.g. DMP) with the policy parameters.
   */
  Rollout(const Eigen::MatrixXd& policy_parameters, const Eigen::MatrixXd& cost_vars);
  
  /** Constructor.
   * \param[in] policy_parameters The parameters of the policy for this rollout
   * \param[in] cost_vars The cost-relevant variables that arose from executing the policy (e.g. DMP) with the policy parameters.
   * \param[in] cost The cost of this rollout. The first element cost[0] should be the total cost. The others may be the individual cost components that consitute the total cost, e.g. cost[0] = cost[1] + cost[2] ...
   */
  Rollout(const Eigen::MatrixXd& policy_parameters, const Eigen::MatrixXd& cost_vars, const Eigen::MatrixXd& cost);
  
  /** Accessor function.
   * \param[in] cost_vars The cost-relevant variables that arose from executing the policy (e.g. DMP) with the policy parameters.
   */
  inline void set_cost_vars(const Eigen::MatrixXd& cost_vars)
  {
    cost_vars_ = cost_vars;
  }
  
  /** Accessor function.
   * \param[in] cost The cost of this rollout. The first element cost[0] should be the total cost. The others may be the individual cost components that consitute the total cost, e.g. cost[0] = cost[1] + cost[2] ...
   */
  inline void set_cost(const Eigen::VectorXd& cost)
  { 
    assert(cost.size()>=1);
    cost_ = cost;
  }
  
  /** Accessor function.
   * \param[out] cost The cost of this rollout. The first element cost[0] should be the total cost. The others may be the individual cost components that consitute the total cost, e.g. cost[0] = cost[1] + cost[2] ...
   */
  inline void cost(Eigen::VectorXd& cost) const
  { 
    cost = Eigen::VectorXd(cost_);
  }
  
  /** Get the (total) cost of the rollout.
   * \return The cost of this rollout.
   */
  double total_cost(void) const;
  
  
  /** Get the number of individual cost components that constitute the final total cost.
   * \return The number of cost components.
   */
  unsigned int getNumberOfCostComponents(void) const;
  
  /**
   * Save a rollout to a directory
   * \param[in] directory Directory to which to write object
   * \param[in] overwrite Overwrite existing files in the directory above (default: false)
   * \return true if saving the Rollout was successful, false otherwise
   */
  bool saveToDirectory(std::string directory, bool overwrite=false) const;
  
private:
  /** The policy parameters. */
  Eigen::MatrixXd policy_parameters_;
  /** The cost-relevant variables resulting from executing the policy. */
  Eigen::MatrixXd cost_vars_;
  /** The cost of this rollout. */
  Eigen::VectorXd cost_;  
  
  /** Give boost serialization access to private members. */  
  friend class boost::serialization::access;
   
  /** Serialize class data members to boost archive. 
   * \param[in] ar Boost archive
   * \param[in] version Version of the class
   * See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/tutorial.html#simplecase
   */
  template<class Archive>
  void serialize(Archive & ar, DmpBbo::Rollout& rollout, const unsigned int version)
  {
    ar & BOOST_SERIALIZATION_NVP(policy_parameters_);
    ar & BOOST_SERIALIZATION_NVP(cost_vars_);
    ar & BOOST_SERIALIZATION_NVP(cost_);
  }
  
};




}
  

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::Rollout,boost::serialization::object_serializable);

#endif


