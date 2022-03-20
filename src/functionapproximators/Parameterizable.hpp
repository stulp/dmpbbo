/**
 * @file   Parameterizable.hpp
 * @brief  Parameterizable class header file.
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

#ifndef PARAMETERIZABLE_H
#define PARAMETERIZABLE_H

#include "eigen_realtime/eigen_realtime_check.hpp" // Include this before Eigen header files

#include "dmpbbo_io/EigenBoostSerialization.hpp"

#include <set>
#include <string>
#include <vector>
#include <eigen3/Eigen/Core>

#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>


namespace DmpBbo {

/** \brief Class for providing access to a model's parameters as a vector.
 *
 * Different function approximators have different types of model parameters. For instance, LWR
 * has the centers and widths of basis functions, along with the slopes of each line segment.
 * Parameterizable::getValues provides a means to access these parameter as one vector.
 *
 * This may be useful for instance when optimizing the model parameters with black-box
 * optimization, which is agnostic about the semantics of the model parameters. 
 */
class Parameterizable {
  
public: 
  
  /** Destructor */
  virtual ~Parameterizable(void) {};
  
  /** Return all the names of the parameter types that can be selected.
   * \param[out] selected_values_labels Names of the parameter types that can be selected
   */
  virtual void getSelectableParameters(std::set<std::string>& selected_values_labels) const = 0;
  
  
  /**
   * Determine which subset of parameters is represented in the vector returned by Parameterizable::getParameterVector
   * 
   * Different function approximators have different types of model parameters. For instance, LWR
   * has the centers and widths of basis functions, along with the slopes of each line segment.
   * Parameterizable::setSelectedParameters provides a means to determine which parameters 
   * should be returned by Parameterizable::getParameterVector, i.e. by calling:
   *   std::set<std::string> selected;
   *   selected.insert("slopes");
   *   model_parameters.setSelectedParameters(selected)
   * \param[in] selected_values_labels The names of the parameters that are selected
   */
  virtual void setSelectedParameters(const std::set<std::string>& selected_values_labels) = 0;
  
  virtual int getParameterVectorSize(void) const = 0;

  /** Set the parameters that are currently selected. 
   * Convenience function that allow only a string to be passed, rather than a set of strings.
   * \param[in] selected The name of the parameters that are selected
   */
  inline void setSelectedParameter(std::string selected)
  {
    std::set<std::string> selected_set;
    selected_set.insert(selected);
    setSelectedParameters(selected_set);    
  }
  
  /**
   * Get the values of the selected parameters in one vector.
   * \param[out] values The selected parameters concatenated in one vector
   * \param[in] normalized Whether to normalize the data or not
   */
  virtual void getParameterVector(Eigen::VectorXd& values, bool normalized=false) const = 0;
  
  /**
   * Get the normalized values of the selected parameters in one vector.
   * \param[out] values The selected parameters concatenated in one vector
   */
  inline void getParameterVectorNormalized(Eigen::VectorXd& values) const {
    getParameterVector(values, true);
  }
  
  /**
   * Set all the values of the selected parameters with one vector.
   * \param[in] values The new values of the selected parameters in one vector
   * \param[in] normalized Whether the data is normalized or not
   */
  virtual void setParameterVector(const Eigen::VectorXd& values, bool normalized=false) = 0;
  
  /**
   * Set all the values of the selected parameters with one vector of normalized values.
   * \param[in] values The new values of the selected parameters in one vector of normalized values.
   */
  inline void setParameterVectorNormalized(const Eigen::VectorXd& values) {
    setParameterVector(values, true);
  }

  

  /** Turn certain modifiers on or off.
   *
   * This can be used to modify exactly what is returned by Parameterizable::getParameterVector(). 
   * For an example, see ModelParametersLWR::setParameterVectorModifierPrivate()
   *
   * This function calls the virtual private function Parameterizable::setParameterVectorModifierPrivate(), which may (but must not be) overridden by subclasses of Parameterizable.
   *
   * \param[in] modifier The name of the modifier
   * \param[in] new_value Whether to turn the modifier on (true) or off (false)
   */
  void setParameterVectorModifier(std::string modifier, bool new_value);
  
  /** The vector (VectorXd) with parameter values can be split into different parts (as vector<VectorXd>; this function specifices the length of each sub-vector.
   * 
   * For instance if the parameter vector is of length 12, getParameterVector(VectorXd) would return a VectorXd of size 12.
   * If you would like these 16 values to be split into 4 VectorXd of length 3, you would set 
   * setVectorLengthsPerDimension([3 3 3 3]).
   * getParameterVector(VectorXd) would still return a VectorXd of size 12, but getParameterVector(std::vector<Eigen::VectorXd>&) would return a std::vector of length 4, with each VectorXd of size 3.
   *
   * This is a convenience function to be able to use vector<VectorXd> instead of VectorXd when getting/setting parameter values.
   *
   * \param[in] lengths_per_dimension The length of each vector in each dimension.
   */
  void setVectorLengthsPerDimension(const Eigen::VectorXi& lengths_per_dimension)
  {
    assert(lengths_per_dimension.sum()==getParameterVectorSize());
    lengths_per_dimension_ = lengths_per_dimension;
  }

  /** Get the specified length of each vector in each dimension.
   * \see setVectorLengthsPerDimension()
   * \return The length of each vector in each dimension.
   */
  Eigen::VectorXi getVectorLengthsPerDimension(void) const
  {
    return lengths_per_dimension_;
  }

private:
  /** Turn certain modifiers on or off, see Parameterizable::setParameterVectorModifier().
   *
   * Parameterizable::setParameterVectorModifierPrivate(), This function may (but must not be) overridden by subclasses of Parameterizable, depending on whether the subclass has modifiers (or not)
   *
   * \param[in] modifier The name of the modifier
   * \param[in] new_value Whether to turn the modifier on (true) or off (false)
   */
  virtual void setParameterVectorModifierPrivate(std::string modifier, bool new_value)
  {
    // Can be overridden by subclasses
  }
  
  /** 
   * \see Parameterizable::setVectorLengthsPerDimension()
   */
  Eigen::VectorXi lengths_per_dimension_;
  
  // Since this is a cached variable, it needs to be mutable so that const functions may change it.
  mutable Eigen::VectorXd parameter_vector_all_initial_;
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
    ar & BOOST_SERIALIZATION_NVP(parameter_vector_all_initial_);
  }

};

}

#endif
