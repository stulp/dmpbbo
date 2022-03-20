/**
 * @file   ModelParameters.hpp
 * @brief  ModelParameters class header file.
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
 
#ifndef MODELPARAMETERS_H
#define MODELPARAMETERS_H

#include "Parameterizable.hpp"

#include <iosfwd>
#include <set>
#include <string>
#include <unordered_map>

#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/unordered_map.hpp>

#include "dmpbbo_io/EigenBoostSerialization.hpp"

namespace DmpBbo {

class UnifiedModel;
  
/** \brief Base class for all model parameters of function approximators
 * \ingroup FunctionApproximators
 */
class ModelParameters : public Parameterizable
{
public:
  
  /** Return a pointer to a deep copy of the ModelParameters object.
   *  \return Pointer to a deep copy
   */
  virtual ModelParameters* clone(void) const = 0;
  
  virtual ~ModelParameters(void) {};

  /** Print to output stream. 
   *
   *  \param[in] output  Output stream to which to write to
   *  \param[in] model_parameters Model-parameters to write
   *  \return    Output stream
   *
   *  \remark Calls virtual function ModelParameters::toString, which must be implemented by
   * subclasses: http://stackoverflow.com/questions/4571611/virtual-operator
   */ 
   friend std::ostream& operator<<(std::ostream& output, const ModelParameters& model_parameters) {
    output << model_parameters.toString();
    return output;
  }
  
  /** Returns a string representation of the object.
   * \return A string representation of the object.
   */
  virtual std::string toString(void) const = 0;
  
  /** The expected dimensionality of the input data.
   * \return Expected dimensionality of the input data
   */
  virtual int getExpectedInputDim(void) const  = 0;
  
  /** The expected dimensionality of the output data.
   * For now, we only consider 1-dimensional output by default.
   * \return Expected dimensionality of the output data
   */
  virtual int getExpectedOutputDim(void) const
  {
    return 1;
  }
  
  /** 
   * Convert these model parameters to unified model parameters.
   * \return Unified model parameter representation (NULL if not implemented for a particular subclass)
   */
  virtual UnifiedModel* toUnifiedModel(void) const = 0;

  void setSelectedParameters(const std::set<std::string>& labels);
  bool isParameterSelected(std::string label) const;
  int getParameterVectorSize(void) const; 
  
protected:
  void checkMinMax(void);
  std::set<std::string> selected_param_labels_;
  std::unordered_map<std::string, double> min_;
  std::unordered_map<std::string, double> max_;
  std::unordered_map<std::string, int> sizes_;

public:
  
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
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Parameterizable);
    ar & BOOST_SERIALIZATION_NVP(selected_param_labels_);
    ar & BOOST_SERIALIZATION_NVP(min_);
    ar & BOOST_SERIALIZATION_NVP(max_);
    ar & BOOST_SERIALIZATION_NVP(sizes_);
  }

};

}

#endif //  #ifndef MODELPARAMETERS_H

