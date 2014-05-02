/**
 * @file   UpdateSummary.hpp
 * @brief  UpdateSummary class header file.
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

#ifndef UPDATESUMMARY_H
#define UPDATESUMMARY_H   

#include <string>
#include <vector>
#include <eigen3/Eigen/Core>

#include <boost/serialization/nvp.hpp>

namespace DmpBbo {
  
// Forward declaration
class DistributionGaussian;

// POD class, http://en.wikipedia.org/wiki/Plain_old_data_structure
/** POD class for storing the information relevant to a distribution update. 
 * Used for logging purposes.
 * This is a "plain old data" class, i.e. all member variables are public
 */
class UpdateSummary {
public:
  /** Distribution before update. */
  DistributionGaussian* distribution;
  /** Samples in the epoch. */
  Eigen::MatrixXd samples;
  /** Cost of the evaluation sample */
  double cost_eval;
  /** Costs of the samples in the epoch. */
  Eigen::VectorXd costs;
  /** Weights of the samples in the epoch, computed from their costs. */
  Eigen::MatrixXd weights;
  /** Distribution after the update. */
  DistributionGaussian* distribution_new;
  
  /** The cost-relevant variables. Only used when Task/TaskSolver approach is used.  */
  Eigen::MatrixXd cost_vars;
  /** The cost-relevant variables for the evaluation. 
      Only used when Task/TaskSolver approach is used.  */
  Eigen::MatrixXd cost_vars_eval;
  
};

/**
 * Save an update summary to a directory
 * \param[in] update_summary Object to write
 * \param[in] directory Directory to which to write object
 * \param[in] overwrite Overwrite existing files in the directory above (default: false)
 * \return true if saving the UpdateSummary was successful, false otherwise
 */
bool saveToDirectory(const UpdateSummary& update_summary, std::string directory, bool overwrite=false);


/**
 * Save a vector of update summaries to a directory
 * \param[in] update_summary Object to write
 * \param[in] directory Directory to which to write object
 * \param[in] overwrite Overwrite existing files in the directory above (default: false)
 * \param[in] only_learning_curve Save only the learning curve (default: false)
 * \return true if saving the UpdateSummary was successful, false otherwise
 */
bool saveToDirectory(const std::vector<UpdateSummary>& update_summaries, std::string directory, bool overwrite=false, bool only_learning_curve=false);

/**
 * Save an update summary to a directory.
 * This version searches directory for subdirectories updateN and writes the data into a new 
 * directory updateN+1 (the actual format is update%05d). E.g. if there are directories:
 * update00000, update00001, and update00002, this function will write to update00003
 * \param[in] update_summary Object to write
 * \param[in] directory Directory to which to write object
 * \return true if saving the UpdateSummary was successful, false otherwise
 */
bool saveToDirectoryNewUpdate(const UpdateSummary& update_summary, std::string directory);

}


/** Serialization function for boost::serialization. */
namespace boost {
namespace serialization {

/** Serialize class data members to boost archive. 
 * \param[in] ar Boost archive
 * \param[in] update_summary UpdateSummary object to serialize.
 * \param[in] version Version of the class
 * See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/tutorial.html#simplecase
 */
template<class Archive>
void serialize(Archive & ar, DmpBbo::UpdateSummary& update_summary, const unsigned int version)
{
  ar & BOOST_SERIALIZATION_NVP(update_summary.distribution);
  ar & BOOST_SERIALIZATION_NVP(update_summary.samples);
  ar & BOOST_SERIALIZATION_NVP(update_summary.cost_eval);
  ar & BOOST_SERIALIZATION_NVP(update_summary.costs);
  ar & BOOST_SERIALIZATION_NVP(update_summary.weights);
  ar & BOOST_SERIALIZATION_NVP(update_summary.distribution_new);
  ar & BOOST_SERIALIZATION_NVP(update_summary.cost_vars);
  ar & BOOST_SERIALIZATION_NVP(update_summary.cost_vars_eval);
}

} // namespace serialization
} // namespace boost


/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::UpdateSummary,boost::serialization::object_serializable);

#endif

