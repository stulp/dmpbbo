/**
 * @file BoostSerializationToString.hpp
 * @brief  Header file to generate strings from boost serialized files.
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

#ifndef BOOSTSERIALIZATIONTOSTRING_HPP
#define BOOSTSERIALIZATIONTOSTRING_HPP

#include <boost/serialization/nvp.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

/** Macro to convert the boost XML serialization of an object into a string.
 * For example, see ExponentialSystem::toString()
 */
#define RETURN_STRING_FROM_BOOST_SERIALIZATION_XML(name) \
  std::stringstream strstr; \
  unsigned int flags = boost::archive::no_header; \
  boost::archive::xml_oarchive oa(strstr,flags); \
  oa << boost::serialization::make_nvp(name, *this); \
  return strstr.str();
  
/** Macro to convert the boost text serialization of an object into a string.
 * For example, see ExponentialSystem::toString()
 */
#define RETURN_STRING_FROM_BOOST_SERIALIZATION_TXT(name) \
  std::stringstream strstr; \
  unsigned int flags = boost::archive::no_header; \
  boost::archive::txt_oarchive oa(strstr,flags); \
  oa << boost::serialization::make_nvp(name, *this); \
  return strstr.str();

#endif        //  #ifndef BOOSTSERIALIZATIONTOSTRING_HPP

