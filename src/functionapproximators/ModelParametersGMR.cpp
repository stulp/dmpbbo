/**
 * @file   ModelParametersGMR.cpp
 * @brief  ModelParametersGMR class source file.
 * @author Thibaut Munzer, Freek Stulp
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
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include "functionapproximators/ModelParametersGMR.hpp"

BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::ModelParametersGMR);

#include "dmpbbo_io/EigenBoostSerialization.hpp"
#include "dmpbbo_io/BoostSerializationToString.hpp"

#include <iostream>
#include <eigen3/Eigen/LU>

using namespace Eigen;
using namespace std;

namespace DmpBbo {

ModelParametersGMR::ModelParametersGMR(std::vector<double> priors,
  std::vector<Eigen::VectorXd> mu_xs, std::vector<Eigen::VectorXd> mu_ys,
  std::vector<Eigen::MatrixXd> sigma_xs, std::vector<Eigen::MatrixXd> sigma_ys,
  std::vector<Eigen::MatrixXd> sigma_y_xs)
:
  priors_(priors),
  mu_xs_(mu_xs),
  mu_ys_(mu_ys),
  sigma_xs_(sigma_xs),
  sigma_ys_(sigma_ys),
  sigma_y_xs_(sigma_y_xs)
{
  
#ifndef NDEBUG // Check for NDEBUG to avoid 'unused variable' warnings for nb_in_dim and nb_out_dim.
  size_t nb_receptive_fields = priors.size();
  assert(nb_receptive_fields>0);
  assert(mu_xs_.size() == nb_receptive_fields);
  assert(mu_ys_.size() == nb_receptive_fields);
  assert(sigma_xs_.size() == nb_receptive_fields);
  assert(sigma_ys_.size() == nb_receptive_fields);
  assert(sigma_y_xs_.size() == nb_receptive_fields);

  int nb_in_dim = getExpectedInputDim();
  for (size_t i = 0; i < nb_receptive_fields; i++)
  {
    assert(mu_xs_[i].size() == nb_in_dim);
    assert(sigma_xs_[i].rows() == nb_in_dim);
    assert(sigma_xs_[i].cols() == nb_in_dim);
    assert(sigma_y_xs_[i].cols() == nb_in_dim);
  }

  int nb_out_dim = mu_ys_[0].size();
  for (size_t i = 0; i < nb_receptive_fields; i++)
  {
    assert(sigma_ys_[i].rows() == nb_out_dim);
    assert(sigma_ys_[i].cols() == nb_out_dim);
    assert(sigma_y_xs_[i].rows() == nb_out_dim);
  }
#endif

  for (size_t i = 0; i < nb_receptive_fields; i++)
    inverted_sigma_xs_.push_back(sigma_y_xs_[i].inverse());

  all_values_vector_size_ = 0;
  
  // NEW REPRESENTATION
  // all_values_vector_size_ += nb_receptive_fields;

  // all_values_vector_size_ += nb_receptive_fields * nb_in_dim;
  // all_values_vector_size_ += nb_receptive_fields * nb_out_dim;

  // all_values_vector_size_ += nb_receptive_fields * nb_in_dim * nb_in_dim;
  // all_values_vector_size_ += nb_receptive_fields * nb_out_dim * nb_out_dim;
  // all_values_vector_size_ += nb_receptive_fields * nb_out_dim * nb_in_dim;  
};

ModelParameters* ModelParametersGMR::clone(void) const
{
  std::vector<double> priors;
  std::vector<VectorXd> mu_xs;
  std::vector<VectorXd> mu_ys;
  std::vector<MatrixXd> sigma_xs;
  std::vector<MatrixXd> sigma_ys;
  std::vector<MatrixXd> sigma_y_xs;

  for (size_t i = 0; i < priors_.size(); i++)
  {
    priors.push_back(priors_[i]);
    mu_xs.push_back(VectorXd(mu_xs_[i]));
    mu_ys.push_back(VectorXd(mu_xs_[i]));
    sigma_xs.push_back(MatrixXd(mu_xs_[i]));
    sigma_ys.push_back(MatrixXd(mu_ys_[i]));
    sigma_y_xs.push_back(MatrixXd(sigma_y_xs_[i]));
  }

  return new ModelParametersGMR(priors, mu_xs, mu_ys, sigma_xs, sigma_ys, sigma_y_xs); 
}

int ModelParametersGMR::getExpectedInputDim(void) const  {
  assert(mu_xs_.size()>0); // This is also checked in the constructor
  return mu_xs_[0].size();
};

template<class Archive>
void ModelParametersGMR::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ModelParameters);
  
  ar & BOOST_SERIALIZATION_NVP(priors_);
  ar & BOOST_SERIALIZATION_NVP(mu_xs_);
  ar & BOOST_SERIALIZATION_NVP(mu_ys_);
  ar & BOOST_SERIALIZATION_NVP(sigma_xs_);
  ar & BOOST_SERIALIZATION_NVP(sigma_ys_);
  ar & BOOST_SERIALIZATION_NVP(sigma_y_xs_);
}


string ModelParametersGMR::toString(void) const 
{
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("ModelParametersGMR");
};


void ModelParametersGMR::getSelectableParameters(set<string>& selected_values_labels) const 
{
  selected_values_labels = set<string>();
  // selected_values_labels.insert("centers");
  // selected_values_labels.insert("priors");
  // selected_values_labels.insert("slopes");
  // selected_values_labels.insert("biases");
  // selected_values_labels.insert("inverse_covars_l");
}

void ModelParametersGMR::getParameterVectorMask(const std::set<std::string> selected_values_labels, VectorXi& selected_mask) const
{
  // selected_mask.resize(getParameterVectorAllSize());
  // selected_mask.fill(0);
  
  // int offset = 0;
  // int size;

  // size = centers_.size() * centers_[0].size();
  // if (selected_values_labels.find("centers")!=selected_values_labels.end())
  //   selected_mask.segment(offset,size).fill(1);
  // offset += size;
  
  // size = priors_.size();
  // if (selected_values_labels.find("priors")!=selected_values_labels.end())
  //   selected_mask.segment(offset,size).fill(2);
  // offset += size;

  // size = slopes_.size() * slopes_[0].rows() * slopes_[0].cols();
  // if (selected_values_labels.find("slopes")!=selected_values_labels.end())
  //   selected_mask.segment(offset,size).fill(3);
  // offset += size;

    
  // size = biases_.size() * biases_[0].size();
  // if (selected_values_labels.find("biases")!=selected_values_labels.end())
  //   selected_mask.segment(offset,size).fill(4);
  // offset += size;

  // size = inverseCovarsL_.size() * (inverseCovarsL_[0].rows() * (inverseCovarsL_[0].cols() + 1))/2;
  // if (selected_values_labels.find("inverse_covars_l")!=selected_values_labels.end())
  //   selected_mask.segment(offset,size).fill(5);
  // offset += size;
    
  // assert(offset == getParameterVectorAllSize());   
}


void ModelParametersGMR::getParameterVectorAll(VectorXd& values) const
{
  // values.resize(getParameterVectorAllSize());
  // int offset = 0;

  // for (size_t i = 0; i < centers_.size(); i++)
  // {
  //   values.segment(offset, centers_[i].size()) = centers_[i];
  //   offset += centers_[i].size();
  // }

  // for (size_t i = 0; i < centers_.size(); i++)
  // {
  //   values[offset] = priors_[i];
  //   offset += 1;
  // }

  // for (size_t i = 0; i < slopes_.size(); i++)
  // {
  //   for (int col = 0; col < slopes_[i].cols(); col++)
  //   {
  //     values.segment(offset, slopes_[i].rows()) = slopes_[i].col(col);
  //     offset += slopes_[i].rows();
  //   }
  // }

  // for (size_t i = 0; i < centers_.size(); i++)
  // {
  //   values.segment(offset, biases_[i].size()) = biases_[i];
  //   offset += biases_[i].size();
  // }

  // for (size_t i = 0; i < inverseCovarsL_.size(); i++)
  // {
  //   for (int row = 0; row < inverseCovarsL_[i].rows(); row++)
  //     for (int col = 0; col < row + 1; col++)
  //     {
  //       values[offset] = inverseCovarsL_[i](row, col);
  //       offset += 1;
  //     }
  // }
  
  // assert(offset == getParameterVectorAllSize());   

};

void ModelParametersGMR::setParameterVectorAll(const VectorXd& values)
{
  // if (all_values_vector_size_ != values.size())
  // {
  //   cerr << __FILE__ << ":" << __LINE__ << ": values is of wrong size." << endl;
  //   return;
  // }
  
  // int offset = 0;

  // for (size_t i = 0; i < centers_.size(); i++)
  // {
  //   centers_[i] = values.segment(offset, centers_[i].size());
  //   offset += centers_[i].size();
  // }
  // for (size_t i = 0; i < centers_.size(); i++)
  // {
  //   priors_[i] = values[offset];
  //   offset += 1;
  // }

  // for (size_t i = 0; i < slopes_.size(); i++)
  // {
  //   for (int col = 0; col < slopes_[i].cols(); col++)
  //   {
  //     slopes_[i].col(col) = values.segment(offset, slopes_[i].rows());
  //     offset += slopes_[i].rows();
  //   }
  // }

  // for (size_t i = 0; i < centers_.size(); i++)
  // {
  //   biases_[i] = values.segment(offset, biases_[i].size());
  //   offset += biases_[i].size();
  // }

  // for (size_t i = 0; i < inverseCovarsL_.size(); i++)
  // {
  //   for (int row = 0; row < inverseCovarsL_[i].rows(); row++)
  //     for (int col = 0; col < row + 1; col++)
  //     {
  //       inverseCovarsL_[i](row, col) = values[offset];
  //       offset += 1;
  //     }
  // }
  
  // assert(offset == getParameterVectorAllSize());

};

}
