/**
 * @file EigenBoostSerialization.hpp
 * @brief  Header file for serialization of Eigen matrices.
 * @author Freek Stulp
 *
 * Implementations are in the tpp file
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

#ifndef EIGENBOOSTSERIALIZATION_HPP
#define EIGENBOOSTSERIALIZATION_HPP

#include <eigen3/Eigen/Core>

#include <boost/serialization/split_free.hpp>
#include <boost/serialization/level.hpp>

namespace Eigen {
/** Convert an Eigen matrix to a string. */
template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
std::string toString(const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & matrix);
}  // namespace Eigen

namespace boost { 
namespace serialization {
  
/** Serialize an Eigen matrix. */
template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline void save(
    Archive & ar, 
    const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & matrix, 
    const unsigned int file_version
);

/** Deserialize an Eigen matrix. */
template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline void load(
    Archive & ar, 
    Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & matrix, 
    const unsigned int file_version
);

/** Serialize an Eigen matrix. */
template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline void serialize(
    Archive & ar,
    Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & t, 
    const unsigned int file_version
){
    split_free(ar, t, file_version); 
}


} // namespace serialization
} // namespace boost

// The next structure is not so relevant for the documentation, and has a very long name. So we 
// exclude it from the documentation with doxygen's "cond"
/// \cond HIDDEN_SYMBOLS

namespace boost {
  namespace serialization {
    template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    /** Implementation level for Eigen matrices.
     * Basically, this tells boost:serialization not to include version information in the output.
     * See also: http://www.boost.org/doc/libs/1_35_0/libs/serialization/doc/traits.html#level
     * This whole thing is needed because BOOST_CLASS_IMPLEMENTATION doesn't work for templated
     * clasess (such as the Eigen Matrix)
     */
    struct implementation_level<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> > 
    {
      /** Typedef for tag */
      typedef mpl::integral_c_tag tag;
      /** Typedef for tag */
      typedef mpl::int_< boost::serialization::object_serializable > type;
      /** Set implementation_level type */
      static const int value = implementation_level::type::value;
    };
  }
}
/// \endcond

#include "EigenBoostSerialization.tpp"

/**
 * \page page_serialization Serialization
 
Serialization and deserialization of objects serves two purposes in this code: 

\li Reading parameter/experiment settings from a file. For instance, you could specify the MetaParameters of a FunctionApproximator in a file, read that file in your program, and train a FunctionApproximator with those MetaParameters. To try different MetaParameters settings you only have to change the input file, without having to recompile.
To be able to edit such parameter files, the files should be human readable. Ideally, it should be JSON, because it is very readable and compact.

\li Saving the results of an experiment. Saving such results to binary files would be more compact, but to ensure that the same format is used for both serialization and deserialization, we use a human-readable format here also.


I considered the following options.

\li Boost property tree: it makes a mess of arrays when saving to JSON

\li Google protobuf: I preferred to not have code generated for me

\li cereal: http://uscilab.github.io/cereal/, external library. Really the best option, but I tried to avoid users having to use non-standard libraries.

\li jsoncpp: Not ideal for serialization, more for read/write of JSON.

In summary, I could not find any libraries that were easy to install and could serialize to/from JSON. Therefore, I went for the second choice, which was XML. Because boost is a standard, and compiles on most platforms, I decided to use boost:: serialization. I consider this to be quite a compromise, because I find boost::serialization quite messy, it is not well documented, and it took me quite a while to get it working properly (especially the registering of derived classes, for which you have to use a wierd combination of macros in the exact right places)

Since I was writing to XML with boost::serialization anyway, many classes implement a toString() method that simply returns the XML code (without a header) that results from serialization. 
The RETURN_STRING_FROM_BOOST_SERIALIZATION_XML macro does all the work for writing to an XML archive and converting it to a string. Having XML as output is not ideal, but it avoid lots of duplicate code. Perhaps boost::serialization will one day be able to write to JSON also, and then this could be used instead. 

I decided to go for string toString(void) instead of ostream& toStream(ostream&), because toString allows you to easily use both the output stream operator  (output << obj.toString()) and printf (printf("%s",obj.toString()), whereas toStream would be much more messy to use in combination with printf (not everyone likes to use the outputstream operator). 

\section sec_boost_serialization_ugliness Boost serialization issues

With boost::serialization, it is possible to serialize classes without a default constuctor with 
http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/serialization.html#constructors
But load_construct_data requires the creation of an object, which we cannot when classes are abstract. It seems no-one knows how to solve this:
\li http://lists.boost.org/boost-users/2005/09/13827.php
\li http://boost.2283326.n4.nabble.com/serialization-Serializing-classes-with-no-default-constructors-td2557921.html

For this reason abstract base classes that are to be serialized with boost must have a default constructor, and may not have const members. This is really a pain, because a serialization library should not enforce such design decisions... But this is the way it is in the code.

\section sec_eigen_boost_serialization Serializing boost matrices

To avoid bloating the serialization of Eigen matrices with lots of XML tags, they are serialized in a special way. The following examples show how Eigen matrices are serialized in XML:

\li Standard matrix:
\code
<m>2X3; 0 0 0; 1 1 1</m>
\endcode

\li Vector:
\code
<m>3X1; 0 0 0</m>
<m>1X3; 0 0 0</m>
\endcode

\li Diagonal matrix (only the diagonal is saved, indicated with D instead of X:
\code
<m>3D3; 1 2 3</m>
\endcode

*/

#endif        //  #ifndef EIGENBOOSTSERIALIZATION_HPP

