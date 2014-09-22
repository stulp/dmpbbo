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
#include "functionapproximators/FunctionApproximator.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::ModelParametersGMR);

#include "functionapproximators/ModelParametersUnified.hpp"

#include "dmpbbo_io/EigenBoostSerialization.hpp"
#include "dmpbbo_io/BoostSerializationToString.hpp"
#include "dmpbbo_io/EigenFileIO.hpp"

#include <iostream>
#include <eigen3/Eigen/LU>

using namespace Eigen;
using namespace std;

namespace DmpBbo {

ModelParametersGMR::ModelParametersGMR(std::vector<double> priors,
  std::vector<Eigen::VectorXd> means_x, std::vector<Eigen::VectorXd> means_y,
  std::vector<Eigen::MatrixXd> covars_x, std::vector<Eigen::MatrixXd> covars_y,
  std::vector<Eigen::MatrixXd> covars_y_x)
:
  priors_(priors),
  means_x_(means_x),
  means_y_(means_y),
  covars_x_(covars_x),
  covars_y_(covars_y),
  covars_y_x_(covars_y_x)
{
  
  size_t n_gaussians = priors.size();

#ifndef NDEBUG // Check for NDEBUG to avoid 'unused variable' warnings for n_dims_in and n_dims_out.
  assert(n_gaussians>0);
  assert(means_x_.size() == n_gaussians);
  assert(means_y_.size() == n_gaussians);
  assert(covars_x_.size() == n_gaussians);
  assert(covars_y_.size() == n_gaussians);
  assert(covars_y_x_.size() == n_gaussians);

  int n_dims_in = getExpectedInputDim();
  for (size_t i = 0; i < n_gaussians; i++)
  {
    assert(means_x_[i].size() == n_dims_in);
    assert(covars_x_[i].rows() == n_dims_in);
    assert(covars_x_[i].cols() == n_dims_in);
    assert(covars_y_x_[i].cols() == n_dims_in);
  }

  int n_dims_out = means_y_[0].size();
  for (size_t i = 0; i < n_gaussians; i++)
  {
    assert(covars_y_[i].rows() == n_dims_out);
    assert(covars_y_[i].cols() == n_dims_out);
    assert(covars_y_x_[i].rows() == n_dims_out);
  }
#endif

  for (unsigned int i=0; i<n_gaussians; i++)
    covars_x_inv_.push_back(covars_x_[i].inverse());

  all_values_vector_size_ = 0;
  
  // NEW REPRESENTATION
  // all_values_vector_size_ += n_gaussians;

  // all_values_vector_size_ += n_gaussians * n_dims_in;
  // all_values_vector_size_ += n_gaussians * n_dims_out;

  // all_values_vector_size_ += n_gaussians * n_dims_in * n_dims_in;
  // all_values_vector_size_ += n_gaussians * n_dims_out * n_dims_out;
  // all_values_vector_size_ += n_gaussians * n_dims_out * n_dims_in;  
};

ModelParameters* ModelParametersGMR::clone(void) const
{
  std::vector<double> priors;
  std::vector<VectorXd> means_x;
  std::vector<VectorXd> means_y;
  std::vector<MatrixXd> covars_x;
  std::vector<MatrixXd> covars_y;
  std::vector<MatrixXd> covars_y_x;

  for (size_t i = 0; i < priors_.size(); i++)
  {
    priors.push_back(priors_[i]);
    means_x.push_back(VectorXd(means_x_[i]));
    means_y.push_back(VectorXd(means_x_[i]));
    covars_x.push_back(MatrixXd(means_x_[i]));
    covars_y.push_back(MatrixXd(means_y_[i]));
    covars_y_x.push_back(MatrixXd(covars_y_x_[i]));
  }

  return new ModelParametersGMR(priors, means_x, means_y, covars_x, covars_y, covars_y_x); 
}

int ModelParametersGMR::getExpectedInputDim(void) const  {
  assert(means_x_.size()>0); // This is also checked in the constructor
  return means_x_[0].size();
};

template<class Archive>
void ModelParametersGMR::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ModelParameters);
  
  ar & BOOST_SERIALIZATION_NVP(priors_);
  ar & BOOST_SERIALIZATION_NVP(means_x_);
  ar & BOOST_SERIALIZATION_NVP(means_y_);
  ar & BOOST_SERIALIZATION_NVP(covars_x_);
  ar & BOOST_SERIALIZATION_NVP(covars_y_);
  ar & BOOST_SERIALIZATION_NVP(covars_y_x_);
  ar & BOOST_SERIALIZATION_NVP(covars_x_inv_);
}


string ModelParametersGMR::toString(void) const 
{
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("ModelParametersGMR");
};

bool ModelParametersGMR::saveGMM(string directory, const vector<VectorXd>& centers, const vector<MatrixXd>& covars, int iter)
{
  for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
  {
    stringstream stream;
    stream << "gmm";
    if (iter>=0)
      stream << "_iter" << setw(2) << setfill('0') << iter;
    stream  << "_mu" << setw(3) << setfill('0') << i_gau << ".txt";
    string filename = stream.str();
    if (!saveMatrix(directory, filename,  centers[i_gau],  true))
      return false;
    //cout << "  filename=" << filename << endl;
    
    stringstream stream2;
    stream2 << "gmm";
    if (iter>=0)
      stream2 << "_iter" << setw(2) << setfill('0') << iter;
    stream2  << "_covar" << setw(3) << setfill('0') << i_gau << ".txt";
    filename = stream2.str();
    //cout << "  filename=" << filename << endl;
    if (!saveMatrix(directory, filename,  covars[i_gau],  true))
      return false;    
  }
  return true;
}

bool ModelParametersGMR::saveGMM(string save_directory, bool overwrite) const
{
  if (save_directory.empty())
    return true;
  
  //MatrixXd inputs;
  //FunctionApproximator::generateInputsGrid(min, max, n_samples_per_dim, inputs);
  //saveMatrix(save_directory,"n_samples_per_dim.txt",n_samples_per_dim,overwrite);

  int n_gaussians = means_x_.size();
  int n_dims_in = means_x_[0].size();
  int n_dims_out = means_y_[0].size();
  int n_dims_gmm = n_dims_in + n_dims_out;
  
  
  std::vector<VectorXd> means(n_gaussians);
  std::vector<MatrixXd> covars(n_gaussians);
  for (int i_gau = 0; i_gau < n_gaussians; i_gau++)
  {
    means[i_gau] = VectorXd(n_dims_gmm);
    means[i_gau].segment(0, n_dims_in) = means_x_[i_gau];
    means[i_gau].segment(n_dims_in, n_dims_out) = means_y_[i_gau];

    covars[i_gau] = MatrixXd(n_dims_gmm,n_dims_gmm);
    covars[i_gau].fill(0);
    covars[i_gau].block(0, 0, n_dims_in, n_dims_in) = covars_x_[i_gau]; 
    covars[i_gau].block(n_dims_in, n_dims_in, n_dims_out, n_dims_out) = covars_y_[i_gau]; 
    covars[i_gau].block(n_dims_in, 0, n_dims_out, n_dims_in) = covars_y_x_[i_gau];
    covars[i_gau].block(0, n_dims_in, n_dims_in, n_dims_out) = covars_y_x_[i_gau].transpose();
  }
  
  saveGMM(save_directory,means,covars);
 
  
  return true;  
}

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


ModelParametersUnified* ModelParametersGMR::toModelParametersUnified(void) const
{
  int n_gaussians = means_x_.size();
    
  // This copying is not necessary. It is just done to show which variable in GMR relates to which
  // variable in the unified model. 
  vector<VectorXd> centers = means_x_;
  vector<MatrixXd> covars  = covars_x_;

  vector<VectorXd> slopes(n_gaussians);
  vector<double> offsets(n_gaussians);
  VectorXd offset_as_vector;
  for (int i_gau=0; i_gau<n_gaussians; i_gau++)
  {
    slopes[i_gau] = covars_y_x_[i_gau] * covars_x_inv_[i_gau];
    offset_as_vector = means_y_[i_gau] - slopes[i_gau]*means_x_[i_gau];
    assert(offset_as_vector.size()==1);
    offsets[i_gau] = offset_as_vector[0];
  }
  

  bool normalized_basis_functions = true;
  bool lines_pivot_at_max_activation = false;

  return new ModelParametersUnified(centers, covars, slopes, offsets, priors_,  normalized_basis_functions,lines_pivot_at_max_activation); 
  
}

}
