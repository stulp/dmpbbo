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
  
  Rollout(const Eigen::MatrixXd& policy_parameters);
  
  Rollout(const Eigen::MatrixXd& policy_parameters, const Eigen::MatrixXd& cost_vars);
  
  // ZZZ
  Rollout(const Eigen::MatrixXd& policy_parameters, const Eigen::MatrixXd& cost_vars, const Eigen::MatrixXd& cost);
  
  inline void set_cost_vars(const Eigen::MatrixXd& cost_vars)
  {
    cost_vars_ = cost_vars;
  }
  
  inline void set_cost(const Eigen::VectorXd& cost)
  { 
    assert(cost.size()>=1);
    cost_ = cost;
  }
  
  inline void cost(Eigen::VectorXd& cost) const
  { 
    cost = Eigen::VectorXd(cost_);
  }
  
  inline double total_cost(void) const
  { 
    if (cost_.size()==0)
      return 0.0; // ZZZ Issue warning
    return cost_[0];
  }
  
  inline int n_cost_components(void) const
  {
    if (cost_.size()>1)
      return cost_.size();
    return 1;
  }
  
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
    
};




}


/** Serialization function for boost::serialization. */
namespace boost {
namespace serialization {

/** Serialize class data members to boost archive. 
 * \param[in] ar Boost archive
 * \param[in] rollout Rollout object to serialize.
 * \param[in] version Version of the class
 * See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/tutorial.html#simplecase
 */
template<class Archive>
void serialize(Archive & ar, DmpBbo::Rollout& rollout, const unsigned int version)
{
  ar & BOOST_SERIALIZATION_NVP(rollout.policy_parameters_);
  ar & BOOST_SERIALIZATION_NVP(rollout.cost_vars_);
  ar & BOOST_SERIALIZATION_NVP(rollout.cost_);
}

} // namespace serialization
} // namespace boost


/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::Rollout,boost::serialization::object_serializable);

#endif

