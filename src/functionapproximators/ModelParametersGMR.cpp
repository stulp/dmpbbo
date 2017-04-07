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

#include "functionapproximators/UnifiedModel.hpp"

#include "dmpbbo_io/EigenBoostSerialization.hpp"
#include "dmpbbo_io/BoostSerializationToString.hpp"
#include "dmpbbo_io/EigenFileIO.hpp"

#include <iostream>
#include <eigen3/Eigen/LU>

using namespace Eigen;
using namespace std;

namespace DmpBbo {

ModelParametersGMR::ModelParametersGMR(std::vector<double> priors,
  std::vector<Eigen::VectorXd> means,
  std::vector<Eigen::MatrixXd> covars, int n_dims_out)
{
  size_t n_gaussians = priors.size();
  assert(n_gaussians>0);
  int n_dims_gmm = means[0].size();
  int n_dims_in = n_dims_gmm - n_dims_out;

#ifndef NDEBUG // Check for NDEBUG to avoid 'unused variable' warnings
  assert(n_dims_out>0);
  assert(n_dims_in>0);
  assert(means.size() == n_gaussians);
  assert(covars.size() == n_gaussians);
  for (size_t i = 0; i < n_gaussians; i++)
  {
    assert(means[i].size() == n_dims_gmm);
    assert(covars[i].cols() == n_dims_gmm);
    assert(covars[i].rows() == n_dims_gmm);
  }
#endif

  priors_ = priors;

  means_x_.resize(n_gaussians);
  means_y_.resize(n_gaussians);
  covars_x_.resize(n_gaussians);
  covars_y_.resize(n_gaussians);
  covars_y_x_.resize(n_gaussians);

  for (size_t i = 0; i < n_gaussians; i++)
  {
    means_x_[i] = means[i].segment(0,n_dims_in);
    means_y_[i] = means[i].segment(n_dims_in,n_dims_out);

    covars_x_[i] = covars[i].block(0,0,n_dims_in,n_dims_in);
    covars_y_[i] = covars[i].block(n_dims_in,n_dims_in,n_dims_out,n_dims_out);
    covars_y_x_[i] = covars[i].block(n_dims_in, 0, n_dims_out, n_dims_in);

  }

  updateCachedMembers();

  all_values_vector_size_ = 0;

  n_observations_ = 0;
}

ModelParametersGMR::ModelParametersGMR(int n_observations, std::vector<double> priors,
  std::vector<Eigen::VectorXd> means,
  std::vector<Eigen::MatrixXd> covars,
  int n_dims_out)
:ModelParametersGMR(priors,means,covars,n_dims_out)
{
    assert(n_observations >= 0);

    n_observations_ = n_observations;
}

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

  int n_dims_out = getExpectedOutputDim();
  for (size_t i = 0; i < n_gaussians; i++)
  {
    assert(covars_y_[i].rows() == n_dims_out);
    assert(covars_y_[i].cols() == n_dims_out);
    assert(covars_y_x_[i].rows() == n_dims_out);
  }
#endif

  updateCachedMembers();

  all_values_vector_size_ = 0;

  n_observations_ = 0;

  // NEW REPRESENTATION
  // all_values_vector_size_ += n_gaussians;

  // all_values_vector_size_ += n_gaussians * n_dims_in;
  // all_values_vector_size_ += n_gaussians * n_dims_out;

  // all_values_vector_size_ += n_gaussians * n_dims_in * n_dims_in;
  // all_values_vector_size_ += n_gaussians * n_dims_out * n_dims_out;
  // all_values_vector_size_ += n_gaussians * n_dims_out * n_dims_in;
}

ModelParametersGMR::ModelParametersGMR(int n_observations, std::vector<double> priors,
  std::vector<Eigen::VectorXd> means_x, std::vector<Eigen::VectorXd> means_y,
  std::vector<Eigen::MatrixXd> covars_x, std::vector<Eigen::MatrixXd> covars_y,
  std::vector<Eigen::MatrixXd> covars_y_x)
:ModelParametersGMR(priors, means_x, means_y, covars_x,covars_y, covars_y_x)
{
    assert(n_observations >= 0);

    n_observations_ = n_observations;
}


void ModelParametersGMR::updateCachedMembers(void)
{
  int n_gaussians = getNumberOfGaussians();
  int n_dims_in = getExpectedInputDim();
  
  covars_x_inv_.resize(n_gaussians);
  mvgd_scale_.resize(n_gaussians);
  for (int i=0; i<n_gaussians; i++)
  {
    covars_x_inv_[i] = covars_x_[i].inverse();
    
    // 1/sqrt((2*pi)^k*|Sigma|)
    double in_sqrt = pow(2*M_PI,n_dims_in)*covars_x_[i].determinant();
    mvgd_scale_[i] = pow(in_sqrt,-0.5);
  }
}

ModelParameters* ModelParametersGMR::clone(void) const
{
  std::vector<double> priors;
  std::vector<VectorXd> means_x;
  std::vector<VectorXd> means_y;
  std::vector<MatrixXd> covars_x;
  std::vector<MatrixXd> covars_y;
  std::vector<MatrixXd> covars_y_x;
  int n_observations = n_observations_;

  for (size_t i = 0; i < priors_.size(); i++)
  {
    priors.push_back(priors_[i]);
    means_x.push_back(VectorXd(means_x_[i]));
    means_y.push_back(VectorXd(means_y_[i]));
    covars_x.push_back(MatrixXd(covars_x_[i]));
    covars_y.push_back(MatrixXd(covars_y_[i]));
    covars_y_x.push_back(MatrixXd(covars_y_x_[i]));
  }

  return new ModelParametersGMR(n_observations, priors, means_x, means_y, covars_x, covars_y, covars_y_x);
}

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
  ar & BOOST_SERIALIZATION_NVP(mvgd_scale_);
}


string ModelParametersGMR::toString(void) const 
{
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("ModelParametersGMR");
};

bool ModelParametersGMR::saveGMM(std::string directory, const std::vector<Eigen::VectorXd>& centers, const std::vector<Eigen::MatrixXd>& covars, bool overwrite, int iter)
{
  for (size_t i_gau = 0; i_gau < centers.size(); i_gau++)
  {
    stringstream stream;
    stream << "gmm";
    if (iter>=0)
      stream << "_iter" << setw(2) << setfill('0') << iter;
    stream  << "_mu" << setw(3) << setfill('0') << i_gau << ".txt";
    string filename = stream.str();
    if (!saveMatrix(directory, filename,  centers[i_gau],  overwrite))
      return false;
    //cout << "  filename=" << filename << endl;
    
    stringstream stream2;
    stream2 << "gmm";
    if (iter>=0)
      stream2 << "_iter" << setw(2) << setfill('0') << iter;
    stream2  << "_covar" << setw(3) << setfill('0') << i_gau << ".txt";
    filename = stream2.str();
    //cout << "  filename=" << filename << endl;
    if (!saveMatrix(directory, filename,  covars[i_gau],  overwrite))
      return false;    
  }
  return true;
}

bool ModelParametersGMR::saveGMM(std::string save_directory, bool overwrite) const
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
  
  saveGMM(save_directory,means,covars,overwrite);
 
  
  return true;  
}

void ModelParametersGMR::toMatrix(Eigen::MatrixXd& gmm_as_matrix) const
{
  int n_gaussians = means_x_.size();
  assert(n_gaussians>0);
  int n_dims_in = means_x_[0].size();
  int n_dims_out = means_y_[0].size();
  int n_dims_gmm = n_dims_in + n_dims_out;
  
  int n_rows = 2; // First row contains n_gaussians, n_output_dims, second row contains n_observations and responsability
  for (int i_gau = 0; i_gau < n_gaussians; i_gau++)
  {
    n_rows += 1; // Add one row for the prior
    n_rows += 1; // Add one row for the mean
    n_rows += n_dims_gmm; // For the covariance matrix 
  }
  
  gmm_as_matrix = MatrixXd::Zero(n_rows,n_dims_gmm);
  
  gmm_as_matrix(0,0) = n_gaussians;
  gmm_as_matrix(0,1) = n_dims_out;
  gmm_as_matrix(1,0) = n_observations_;
  
  VectorXd mean = VectorXd(n_dims_gmm);
  MatrixXd covar = MatrixXd(n_dims_gmm,n_dims_gmm);
  int cur_row = 2;
  for (int i_gau = 0; i_gau < n_gaussians; i_gau++)
  {
    mean.segment(0, n_dims_in) = means_x_[i_gau];
    mean.segment(n_dims_in, n_dims_out) = means_y_[i_gau];

    covar.block(0, 0, n_dims_in, n_dims_in) = covars_x_[i_gau]; 
    covar.block(n_dims_in, n_dims_in, n_dims_out, n_dims_out) = covars_y_[i_gau]; 
    covar.block(n_dims_in, 0, n_dims_out, n_dims_in) = covars_y_x_[i_gau];
    covar.block(0, n_dims_in, n_dims_in, n_dims_out) = covars_y_x_[i_gau].transpose();
    
    gmm_as_matrix(cur_row,0) = priors_[i_gau];
    gmm_as_matrix.row(cur_row+1) = mean;
    gmm_as_matrix.block(cur_row+2,0,n_dims_gmm,n_dims_gmm) = covar;
    
    cur_row += 1 + 1 + n_dims_gmm;
  }  
}

ModelParametersGMR* ModelParametersGMR::fromMatrix(const MatrixXd& gmm_matrix)
{
  int n_dims_gmm = gmm_matrix.cols();
  int n_rows = gmm_matrix.rows();
  assert(n_dims_gmm>1);
  assert(n_rows>0);
  
  int n_gaussians = gmm_matrix(0,0);
  int n_dims_out = gmm_matrix(0,1); 
  int n_observations = gmm_matrix(1,0);

  assert(n_rows == (2+ (n_gaussians*(1+1+n_dims_gmm))));
  
  vector<double> priors(n_gaussians);
  vector<VectorXd> means(n_gaussians);
  vector<MatrixXd> covars(n_gaussians);
  
  int cur_row = 2;
  for (int i_gau = 0; i_gau < n_gaussians; i_gau++)
  {
    priors[i_gau] = gmm_matrix(cur_row,0);
    
    means[i_gau] = gmm_matrix.row(cur_row+1);
   
    covars[i_gau] = gmm_matrix.block(cur_row+2,0,n_dims_gmm,n_dims_gmm);
    
    cur_row += 1 + 1 + n_dims_gmm;
  }
    
  return new ModelParametersGMR(n_observations,priors,means,covars,n_dims_out);
}

bool ModelParametersGMR::saveGMMToMatrix(std::string filename, bool overwrite) const
{
  if (filename.empty())
    return true;

  MatrixXd gmm_as_matrix;
  toMatrix(gmm_as_matrix);
  
  if (!saveMatrix(filename, gmm_as_matrix,  overwrite))
    return false;    
    
  return true;  
}

ModelParametersGMR* ModelParametersGMR::loadGMMFromMatrix(std::string filename)
{
  MatrixXd gmm_matrix;
  if (!loadMatrix(filename, gmm_matrix))
    return NULL;
  
  return ModelParametersGMR::fromMatrix(gmm_matrix);
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


UnifiedModel* ModelParametersGMR::toUnifiedModel(void) const
{
  int n_gaussians = means_x_.size();
    
  // This copying is not necessary. It is just done to show which variable in GMR relates to which
  // variable in the unified model. 
  vector<VectorXd> centers = means_x_;
  vector<MatrixXd> covars  = covars_x_;

  vector<VectorXd> slopes(n_gaussians);
  vector<double> offsets(n_gaussians);
  VectorXd rest;
  for (int i_gau=0; i_gau<n_gaussians; i_gau++)
  {
    slopes[i_gau] = (covars_y_x_[i_gau] * covars_x_inv_[i_gau]).transpose();
    
    assert(means_y_[i_gau].size()==1); // Only works for 1D y output for now
    offsets[i_gau] = means_y_[i_gau][0] - slopes[i_gau].dot(means_x_[i_gau]);
  }
  

  bool normalized_basis_functions = true;
  bool lines_pivot_at_max_activation = false;

  return new UnifiedModel(centers, covars, slopes, offsets, priors_,  normalized_basis_functions,lines_pivot_at_max_activation); 
  
}

}
