/**
 * @file   MetaParametersLWR.cpp
 * @brief  MetaParametersLWR class source file.
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

#include <boost/serialization/export.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include "functionapproximators/MetaParametersLWR.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::MetaParametersLWR);


#include "dmpbbo_io/BoostSerializationToString.hpp"
#include "dmpbbo_io/EigenBoostSerialization.hpp"

#include <iostream>
#include <unordered_map>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/vector.hpp>



using namespace Eigen;
using namespace std;

namespace DmpBbo {

MetaParametersLWR::MetaParametersLWR(int expected_input_dim, const std::vector<Eigen::VectorXd>& centers_per_dim, double intersection_height, bool asymmetric_kernels)
:
  MetaParameters(expected_input_dim),
  n_bfs_per_dim_(VectorXi::Zero(0)),
  centers_per_dim_(centers_per_dim),
  intersection_height_(intersection_height),
  asymmetric_kernels_(asymmetric_kernels)
{
  assert(expected_input_dim==(int)centers_per_dim_.size());
  for (unsigned int dd=0; dd<centers_per_dim_.size(); dd++)
    assert(centers_per_dim_[dd].size()>0);
  assert(intersection_height_>0 && intersection_height_<1);
}
  
MetaParametersLWR::MetaParametersLWR(int expected_input_dim, const Eigen::VectorXi& n_bfs_per_dim, double intersection_height, bool asymmetric_kernels) 
:
  MetaParameters(expected_input_dim),
  n_bfs_per_dim_(n_bfs_per_dim),
  centers_per_dim_(std::vector<Eigen::VectorXd>(0)),
  intersection_height_(intersection_height),
  asymmetric_kernels_(asymmetric_kernels)
{
  assert(expected_input_dim==n_bfs_per_dim_.size());
  for (int dd=0; dd<n_bfs_per_dim_.size(); dd++)
    assert(n_bfs_per_dim_[dd]>0);
  assert(intersection_height_>0 && intersection_height_<1);
};

MetaParametersLWR::MetaParametersLWR(int expected_input_dim, int n_bfs, double intersection_height, bool asymmetric_kernels) 
:
  MetaParameters(expected_input_dim),
  n_bfs_per_dim_(VectorXi::Constant(1,n_bfs)),
  centers_per_dim_(std::vector<Eigen::VectorXd>(0)),
  intersection_height_(intersection_height),
  asymmetric_kernels_(asymmetric_kernels)
{
  assert(expected_input_dim==n_bfs_per_dim_.size());
  for (int dd=0; dd<n_bfs_per_dim_.size(); dd++)
    assert(n_bfs_per_dim_[dd]>0);
  assert(intersection_height_>0 && intersection_height_<1);
};

MetaParametersLWR* MetaParametersLWR::clone(void) const
{
  MetaParametersLWR* cloned;
  if (centers_per_dim_.size()>0)
    cloned =  new MetaParametersLWR(getExpectedInputDim(),centers_per_dim_,intersection_height_,asymmetric_kernels_);
  else
    cloned =  new MetaParametersLWR(getExpectedInputDim(),n_bfs_per_dim_,intersection_height_,asymmetric_kernels_);
  
  return cloned;
}

/** \todo Document this rather complex function */
void MetaParametersLWR::getCentersAndWidths(const VectorXd& min, const VectorXd& max, Eigen::MatrixXd& centers, Eigen::MatrixXd& widths) const
{
  int n_dims = getExpectedInputDim();
  assert(min.size()==n_dims);
  assert(max.size()==n_dims);
    
  vector<VectorXd> centers_per_dim_local(n_dims); 
  if (!centers_per_dim_.empty())
  {
    centers_per_dim_local = centers_per_dim_;
  }
  else
  {
    // Centers are not know yet, compute them based on min and max
    for (int i_dim=0; i_dim<n_dims; i_dim++)
      centers_per_dim_local[i_dim] = VectorXd::LinSpaced(n_bfs_per_dim_[i_dim],min[i_dim],max[i_dim]);
  }
  
  // Determine the widths from the centers (separately for each dimension)
  vector<VectorXd> widths_per_dim_local(n_dims); 
  for (int i_dim=0; i_dim<n_dims; i_dim++)
  {
    VectorXd cur_centers = centers_per_dim_local[i_dim]; // Abbreviation for convenience
    int n_centers = cur_centers.size();
    VectorXd cur_widths(n_centers);

    if (n_centers==1)
    {
      cur_widths[0] = 1.0;
    }
    else
    {
      // Consider two neighbouring basis functions, exp(-0.5(x-c0)^2/w^2) and exp(-0.5(x-c1)^2/w^2)
      // Assuming the widths are the same for both, they are certain to intersect at x = 0.5(c0+c1)
      // And we want the activation at x to be 'intersection'. So
      //            y = exp(-0.5(x-c0)^2/w^2)
      // intersection = exp(-0.5((0.5(c0+c1))-c0)^2/w^2)
      // intersection = exp(-0.5((0.5*c1-0.5*c0)^2/w^2))
      // intersection = exp(-0.5((0.5*(c1-c0))^2/w^2))
      // intersection = exp(-0.5(0.25*(c1-c0)^2/w^2))
      // intersection = exp(-0.125((c1-c0)^2/w^2))
      //            w = sqrt((c1-c0)^2/-8*ln(intersection))
      for (int cc=0; cc<n_centers-1; cc++)
      {
        double w = sqrt(pow(cur_centers[cc+1]-cur_centers[cc],2)/(-8*log(intersection_height_)));
        cur_widths[cc] = w;
      }
      cur_widths[n_centers-1] = cur_widths[n_centers-2];
    }
    widths_per_dim_local[i_dim] = cur_widths;
  }

  VectorXd digit_max(n_dims);
  int n_centers = 1;
  for (int i_dim=0; i_dim<n_dims; i_dim++)
  {
    n_centers = n_centers*centers_per_dim_local[i_dim].size();
    digit_max[i_dim] = centers_per_dim_local[i_dim].size();
  }
  VectorXi digit = VectorXi::Zero(n_dims);
  
  centers.resize(n_centers,n_dims);
  widths.resize(n_centers,n_dims);
  int i_center=0;

  while (digit[0]<digit_max(0))
  {
    for (int i_dim=0; i_dim<n_dims; i_dim++)
    {
      centers(i_center,i_dim) = centers_per_dim_local[i_dim][digit[i_dim]];
      widths(i_center,i_dim) = widths_per_dim_local[i_dim][digit[i_dim]];
    }
    i_center++;
  
    // Increment last digit by one
    digit[n_dims-1]++;
    for (int i_dim=n_dims-1; i_dim>0; i_dim--)
    {
      if (digit[i_dim]>=digit_max[i_dim])
      {
        digit[i_dim] = 0;
        digit[i_dim-1]++;
      }
    }
  }
  
}

template<class Archive>
void MetaParametersLWR::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(MetaParameters);

  ar & BOOST_SERIALIZATION_NVP(n_bfs_per_dim_);
  ar & BOOST_SERIALIZATION_NVP(centers_per_dim_);
  ar & BOOST_SERIALIZATION_NVP(intersection_height_);
  ar & BOOST_SERIALIZATION_NVP(asymmetric_kernels_);
}

string MetaParametersLWR::toString(void) const
{
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("MetaParametersLWR");
}

}
