/**
 * @file   UnifiedModel.cpp
 * @brief  UnifiedModel class source file.
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
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include "functionapproximators/UnifiedModel.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::UnifiedModel);

#include "functionapproximators/FunctionApproximator.hpp"
#include "functionapproximators/BasisFunction.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"
#include "dmpbbo_io/BoostSerializationToString.hpp"
#include "dmpbbo_io/EigenBoostSerialization.hpp"

#include <iostream>
#include <fstream>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/LU>


using namespace std;
using namespace Eigen;

namespace DmpBbo {

UnifiedModel::UnifiedModel(const Eigen::MatrixXd& centers, const Eigen::MatrixXd& widths, const Eigen::VectorXd& weights, bool normalized_basis_functions, bool lines_pivot_at_max_activation) 
{
  int n_basis_functions = centers.rows();
  int n_dims = centers.cols();
  
  assert(n_basis_functions==widths.rows());
  assert(n_dims           ==widths.cols());
  assert(n_basis_functions==weights.size());
  
  centers_.resize(n_basis_functions);
  covars_.resize(n_basis_functions);
  slopes_.resize(n_basis_functions);
  offsets_.resize(n_basis_functions);
  priors_.resize(n_basis_functions);
  
  for (int i=0; i<n_basis_functions; i++)
  {
    centers_[i] = centers.row(i);
    covars_[i] = widths.row(i).asDiagonal();
    covars_[i] = covars_[i].array().square();
    offsets_[i] = weights[i];
    
    slopes_[i] = VectorXd::Zero(n_dims);
    priors_[i] = 1.0;
  }
  
  normalized_basis_functions_ = normalized_basis_functions;
  lines_pivot_at_max_activation_ = lines_pivot_at_max_activation;
  slopes_as_angles_ = false;
 
  cosine_basis_functions_ = false;
  
  initializeAllValuesVectorSize();
};

UnifiedModel::UnifiedModel(const Eigen::MatrixXd& centers, const Eigen::MatrixXd& widths, const Eigen::MatrixXd& slopes, const Eigen::VectorXd& offsets, bool normalized_basis_functions, bool lines_pivot_at_max_activation)
{
  int n_basis_functions = centers.rows();
#ifndef NDEBUG // Variables below are only required for asserts; check for NDEBUG to avoid warnings.
  int n_dims = centers.cols();
#endif 

  assert(n_basis_functions==widths.rows());
  assert(n_dims           ==widths.cols());
  assert(n_basis_functions==slopes.rows());
  assert(n_dims           ==slopes.cols());
  assert(n_basis_functions==offsets.size());
  
  centers_.resize(n_basis_functions);
  covars_.resize(n_basis_functions);
  slopes_.resize(n_basis_functions);
  offsets_.resize(n_basis_functions);
  priors_.resize(n_basis_functions);
  
  for (int i=0; i<n_basis_functions; i++)
  {
    centers_[i] = centers.row(i);
    covars_[i] = widths.row(i).asDiagonal();
    covars_[i] = covars_[i].array().square();
    slopes_[i] = slopes.row(i);
    offsets_[i] = offsets[i];
    
    priors_[i] = 1.0;
  }
  
  normalized_basis_functions_ = normalized_basis_functions;
  lines_pivot_at_max_activation_ = lines_pivot_at_max_activation;
  slopes_as_angles_ = false;
  
  cosine_basis_functions_ = false;

  initializeAllValuesVectorSize();  
}


UnifiedModel::UnifiedModel(const Eigen::MatrixXd& angular_frequencies, const Eigen::VectorXd& phases, const Eigen::VectorXd& weights)
{
  int n_basis_functions = angular_frequencies.rows();
  int n_dims = angular_frequencies.cols();
  
  assert(n_basis_functions==phases.size());
  assert(n_basis_functions==weights.size());
  
  centers_.resize(n_basis_functions); // phase
  covars_.resize(n_basis_functions);  // angular_frequencies
  slopes_.resize(n_basis_functions);  
  offsets_.resize(n_basis_functions); // weights
  priors_.resize(n_basis_functions);  
  
  for (int i=0; i<n_basis_functions; i++)
  {
    centers_[i] = VectorXd(1);
    centers_[i][0] = phases[i];
    
    covars_[i] = MatrixXd(1,n_dims);
    covars_[i] = angular_frequencies.row(i);

    offsets_[i] = weights[i];
    
    slopes_[i] = VectorXd::Zero(n_dims);
    priors_[i] = 1.0;
  }

  // These aren't relevant for cosing basis functions
  normalized_basis_functions_ = false;
  lines_pivot_at_max_activation_ = false;
  slopes_as_angles_ = false;
  
  cosine_basis_functions_ = true;
  
  initializeAllValuesVectorSize();
}

UnifiedModel::UnifiedModel(
  const std::vector<Eigen::VectorXd>& centers, // n_centers X n_dims
  const std::vector<Eigen::MatrixXd>& covars,  // n_centers X n_dims X n_dims
  const std::vector<Eigen::VectorXd>& slopes, // n_centers X n_dims
  const std::vector<double>& offsets,          // n_centers X 1
  const std::vector<double>& priors,           // n_centers X 1              
  bool normalized_basis_functions, 
  bool lines_pivot_at_max_activation)
:
  centers_(centers),
  covars_(covars),
  slopes_(slopes), 
  offsets_(offsets),
  priors_(priors), 
  normalized_basis_functions_(normalized_basis_functions),
  lines_pivot_at_max_activation_(lines_pivot_at_max_activation),
  slopes_as_angles_(false)
{
  cosine_basis_functions_ = false;

  initializeAllValuesVectorSize();
}

void UnifiedModel::initializeAllValuesVectorSize(void)
{
  
  all_values_vector_size_ = 0;
  int n_basis_functions = centers_.size();
  if (n_basis_functions>0)
  {
    all_values_vector_size_ += centers_.size()*centers_[0].size();
    all_values_vector_size_ += covars_.size() *covars_[0].cols();
    all_values_vector_size_ += offsets_.size();
    all_values_vector_size_ += slopes_.size() *slopes_[0].size();  
    all_values_vector_size_ += priors_.size();  
  }
}



UnifiedModel* UnifiedModel::clone(void) const {
  return new UnifiedModel(centers_,covars_,slopes_,offsets_,priors_,normalized_basis_functions_,lines_pivot_at_max_activation_); 
}


void UnifiedModel::set_lines_pivot_at_max_activation(bool lines_pivot_at_max_activation)
{
  // If no change, just return
  if (lines_pivot_at_max_activation_ == lines_pivot_at_max_activation)
    return;

  //cout << "________________" << endl;
  //cout << centers_.transpose() << endl;  
  //cout << slopes_.transpose() << endl;  
  //cout << offsets_.transpose() << endl;  
  //cout << "centers_ = " << centers_.rows() << "X" << centers_.cols() << endl;
  //cout << "slopes_ = " << slopes_.rows() << "X" << slopes_.cols() << endl;
  //cout << "offsets_ = " << offsets_.rows() << "X" << offsets_.cols() << endl;

  // If you pivot lines around the point when the basis function has maximum activation (i.e.
  // at the center of the Gaussian), you must compute the new offset corresponding to this
  // slope, and vice versa    
  int n_lines = centers_.size();
  VectorXd ac(n_lines); // slopes*centers
  for (int i_line=0; i_line<n_lines; i_line++)
  {
    ac[i_line] = slopes_[i_line].dot(centers_[i_line]);
    
    if (lines_pivot_at_max_activation)
    {
      // Representation was "y = ax + b", now it will be "y = a(x-c) + b^new" 
      // Since "y = ax + b" can be rewritten as "y = a(x-c) + (b+ac)", we know that "b^new = (ac+b)"
      offsets_[i_line] = offsets_[i_line] + ac[i_line];
    }
    else
    {
      // Representation was "y = a(x-c) + b", now it will be "y = ax + b^new" 
      // Since "y = a(x-c) + b" can be rewritten as "y = ax + (b-ac)", we know that "b^new = (b-ac)"
      offsets_[i_line] = offsets_[i_line] - ac[i_line];
    } 
  }
  
  //cout << offsets_.transpose() << endl;  
  //cout << "offsets_ = " << offsets_.rows() << "X" << offsets_.cols() << endl;
  
  lines_pivot_at_max_activation_ = lines_pivot_at_max_activation;
}

void UnifiedModel::set_slopes_as_angles(bool slopes_as_angles)
{
  slopes_as_angles_ = slopes_as_angles;
  cerr << __FILE__ << ":" << __LINE__ << ":";
  cerr << "Not implemented yet!!!" << endl;
  slopes_as_angles_ = false;
}




void UnifiedModel::getLines(const MatrixXd& inputs, MatrixXd& lines) const
{
  int n_time_steps = inputs.rows();

  //cout << "centers_ = " << centers_.rows() << "X" << centers_.cols() << endl;
  //cout << "slopes_ = " << slopes_.rows() << "X" << slopes_.cols() << endl;
  //cout << "offsets_ = " << offsets_.rows() << "X" << offsets_.cols() << endl;
  //cout << "inputs = " << inputs.rows() << "X" << inputs.cols() << endl;
  
  // Compute values along lines for each time step  
  // Line representation is "y = ax + b"
  int n_lines = centers_.size();
  lines.resize(n_time_steps,n_lines);
  for (int i_line=0; i_line<n_lines; i_line++)
  {
    lines.col(i_line) = inputs*slopes_[i_line];
    lines.col(i_line).array() += offsets_[i_line];
  
    if (lines_pivot_at_max_activation_)
    {
      // Line representation is "y = a(x-c) + b", which is  "y = ax - ac + b"
      // Therefore, we still have to subtract "ac"
      double ac = slopes_[i_line].dot(centers_[i_line]);
      lines.col(i_line).array() -= ac;
    }
  }
  //cout << "lines = " << lines.rows() << "X" << lines.cols() << endl;
}
  
void UnifiedModel::evaluate(const MatrixXd& inputs, MatrixXd& output) const
{
  
  MatrixXd lines;
  getLines(inputs, lines);

  // Weight the values for each line with the normalized basis function activations  
  MatrixXd activations;
  kernelActivations(inputs,activations);
  
  output = (lines.array()*activations.array()).rowwise().sum();
}

/*
void UnifiedModel::kernelActivationsSymmetric(const MatrixXd& centers, const MatrixXd& widths, const MatrixXd& inputs, MatrixXd& kernel_activations)
{
  cout << __FILE__ << ":" << __LINE__ << ":Here" << endl;
  // Check and set sizes
  // centers     = n_basis_functions x n_dim
  // widths      = n_basis_functions x n_dim
  // inputs      = n_samples         x n_dim
  // activations = n_samples         x n_basis_functions
  int n_basis_functions = centers.rows();
  int n_samples         = inputs.rows();
  int n_dims            = centers.cols();
  assert( (n_basis_functions==widths.rows()) & (n_dims==widths.cols()) ); 
  assert( (n_samples==inputs.rows()        ) & (n_dims==inputs.cols()) ); 
  kernel_activations.resize(n_samples,n_basis_functions);  


  VectorXd center, width;
  for (int bb=0; bb<n_basis_functions; bb++)
  {
    center = centers.row(bb);
    width  = widths.row(bb);

    // Here, we compute the values of a (unnormalized) multi-variate Gaussian:
    //   activation = exp(-0.5*(x-mu)*Sigma^-1*(x-mu))
    // Because Sigma is diagonal in our case, this simplifies to
    //   activation = exp(\sum_d=1^D [-0.5*(x_d-mu_d)^2/Sigma_(d,d)]) 
    //              = \prod_d=1^D exp(-0.5*(x_d-mu_d)^2/Sigma_(d,d)) 
    // This last product is what we compute below incrementally
    
    kernel_activations.col(bb).fill(1.0);
    for (int i_dim=0; i_dim<n_dims; i_dim++)
    {
      kernel_activations.col(bb).array() *= exp(-0.5*pow(inputs.col(i_dim).array()-center[i_dim],2)/(width[i_dim]*width[i_dim])).array();
    }
  }
}
*/

void UnifiedModel::kernelActivations(const MatrixXd& inputs, MatrixXd& kernel_activations) const
{
  if (caching_)
  {
    // If the cached inputs matrix has the same size as the one now requested
    //     (this also takes care of the case when inputs_cached is empty and need to be initialized)
    if ( inputs.rows()==inputs_cached_.rows() && inputs.cols()==inputs_cached_.cols() )
    {
      // And they have the same values
      if ( (inputs.array()==inputs_cached_.array()).all() )
      {
        // Then you can return the cached values
        kernel_activations = kernel_activations_cached_;
        return;
      }
    }
  }

  // Cache could not be used, actually do the work
  if (cosine_basis_functions_)
  {
    // centers_.resize(n_basis_functions); // phase
    // covars_.resize(n_basis_functions);  // angular_frequencies
    // slopes_.resize(n_basis_functions);  
    // offsets_.resize(n_basis_functions); // weights
    // priors_.resize(n_basis_functions);  
    BasisFunction::Cosine::activations(covars_,centers_,inputs,kernel_activations);
  }
  else
  {
    BasisFunction::Gaussian::activations(centers_,covars_,priors_,inputs,kernel_activations,normalized_basis_functions_);
  }

  if (caching_)
  {
    // Cache the current results now.  
    inputs_cached_ = inputs;
    kernel_activations_cached_ = kernel_activations;
  }
  
}

template<class Archive>
void UnifiedModel::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Parameterizable);

  ar & BOOST_SERIALIZATION_NVP(centers_);
  ar & BOOST_SERIALIZATION_NVP(covars_);
  ar & BOOST_SERIALIZATION_NVP(slopes_);
  ar & BOOST_SERIALIZATION_NVP(offsets_);
  ar & BOOST_SERIALIZATION_NVP(priors_);
  ar & BOOST_SERIALIZATION_NVP(normalized_basis_functions_);
  ar & BOOST_SERIALIZATION_NVP(lines_pivot_at_max_activation_);
  ar & BOOST_SERIALIZATION_NVP(slopes_as_angles_);
  ar & BOOST_SERIALIZATION_NVP(all_values_vector_size_);
  ar & BOOST_SERIALIZATION_NVP(caching_);
}

string UnifiedModel::toString(void) const
{
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("UnifiedModel");
}

void UnifiedModel::getSelectableParameters(set<string>& selected_values_labels) const 
{
  selected_values_labels = set<string>();
  selected_values_labels.insert("centers");
  selected_values_labels.insert("widths");
  selected_values_labels.insert("offsets");
  selected_values_labels.insert("slopes");
  selected_values_labels.insert("priors");
}


void UnifiedModel::getParameterVectorMask(const std::set<std::string> selected_values_labels, VectorXi& selected_mask) const
{

  selected_mask.resize(getParameterVectorAllSize());
  selected_mask.fill(0);
  
  int offset = 0;
  int size;
  
  // Centers
  size = centers_.size()*centers_[0].size();
  if (selected_values_labels.find("centers")!=selected_values_labels.end())
    selected_mask.segment(offset,size).fill(1);
  offset += size;
  
  // Widths
  size = covars_.size()*covars_[0].cols();
  if (selected_values_labels.find("widths")!=selected_values_labels.end())
    selected_mask.segment(offset,size).fill(2);
  offset += size;
  
  // Offsets
  size = offsets_.size();
  if (selected_values_labels.find("offsets")!=selected_values_labels.end())
    selected_mask.segment(offset,size).fill(3);
  offset += size;

  // Slopes
  size = slopes_.size()*slopes_[0].size();
  if (selected_values_labels.find("slopes")!=selected_values_labels.end())
    selected_mask.segment(offset,size).fill(4);
  offset += size;

  assert(offset == getParameterVectorAllSize());   
}

void UnifiedModel::getParameterVectorAll(VectorXd& values) const
{
  values.resize(getParameterVectorAllSize());
  int offset = 0;
  int n_basis_functions = centers_.size();
  int n_dims = getExpectedInputDim();
  
  for (int i_bfs=0; i_bfs<n_basis_functions; i_bfs++)
  {
    values.segment(offset,n_dims) = centers_[i_bfs];              offset += n_dims;
    values.segment(offset,n_dims) = covars_[i_bfs].diagonal();    offset += n_dims;
    values.segment(offset,n_dims) = slopes_[i_bfs];               offset += n_dims;
    values[offset]                = offsets_[i_bfs];              offset += 1;
    values[offset]                = priors_[i_bfs];               offset += 1;
  }
  
  /*
  Dead code. But kept in for reference in case slopes_as_angles_ will be implemented
  VectorXd cur_slopes;
  for (int i_dim=0; i_dim<slopes_.cols(); i_dim++)
  {
    cur_slopes = slopes_.col(i_dim);
    if (slopes_as_angles_)
    {
      // cur_slopes is a slope, but the values vector expects the angle with the x-axis. Do the 
      // conversion here.
      for (int ii=0; ii<cur_slopes.size(); ii++)
        cur_slopes[ii] = atan2(cur_slopes[ii],1.0);
    }
    
    values.segment(offset,slopes_.rows()) = cur_slopes;
    offset += slopes_.rows();
  }
    */
  
  assert(offset == getParameterVectorAllSize());   
};

void UnifiedModel::setParameterVectorAll(const VectorXd& values) {

  if (all_values_vector_size_ != values.size())
  {
    cerr << __FILE__ << ":" << __LINE__ << ": values is of wrong size." << endl;
    return;
  }

  int offset = 0;
  int n_basis_functions = centers_.size();
  int n_dims = getExpectedInputDim();
  
  for (int i_bfs=0; i_bfs<n_basis_functions; i_bfs++)
  {
    VectorXd cur_center = values.segment(offset,n_dims);
    // If the centers change, the cache for normalizedKernelActivations() must be cleared,
    // because this function will return different values for different centers
    if ( !(centers_[i_bfs].array() == cur_center.array()).all() )
      clearCache();
    
    centers_[i_bfs]           = values.segment(offset,n_dims) ;   offset += n_dims;
    
    VectorXd cur_width = values.segment(offset,n_dims);
    // If the centers change, the cache for normalizedKernelActivations() must be cleared,
    // because this function will return different values for different centers
    if ( !(covars_[i_bfs].diagonal().array() == cur_width.array()).all() )
      clearCache();
    
    covars_[i_bfs].diagonal() = values.segment(offset,n_dims) ;   offset += n_dims;
    
    
    // Cache must not be cleared, because normalizedKernelActivations() returns the same values.
    slopes_[i_bfs]            = values.segment(offset,n_dims) ;   offset += n_dims;
    offsets_[i_bfs]           = values[offset]                ;   offset += 1;
    priors_[i_bfs]            = values[offset]                ;   offset += 1;
  }
  
  /*
  int offset = 0;
  int size = centers_.rows();
  int n_dims = centers_.cols();
  for (int i_dim=0; i_dim<n_dims; i_dim++)
  {
    // If the centers change, the cache for normalizedKernelActivations() must be cleared,
    // because this function will return different values for different centers
    if ( !(centers_.col(i_dim).array() == values.segment(offset,size).array()).all() )
      clearCache();
    
    centers_.col(i_dim) = values.segment(offset,size);
    offset += size;
  }
  for (int i_dim=0; i_dim<n_dims; i_dim++)
  {
    // If the centers change, the cache for normalizedKernelActivations() must be cleared,
    // because this function will return different values for different centers
    if ( !(covars_.col(i_dim).array() == values.segment(offset,size).array()).all() )
      clearCache();
    
    covars_.col(i_dim) = values.segment(offset,size);
    offset += size;
  }

  offsets_ = values.segment(offset,size);
  offset += size;
  // Cache must not be cleared, because normalizedKernelActivations() returns the same values.

  MatrixXd old_slopes = slopes_;
  for (int i_dim=0; i_dim<n_dims; i_dim++)
  {
    slopes_.col(i_dim) = values.segment(offset,size);
    offset += size;
    // Cache must not be cleared, because normalizedKernelActivations() returns the same values.
  }
*/
  assert(offset == getParameterVectorAllSize());   
};

bool UnifiedModel::saveGridData(const VectorXd& min, const VectorXd& max, const VectorXi& n_samples_per_dim, string save_directory, bool overwrite) const
{
  if (save_directory.empty())
    return true;  
 
#ifndef NDEBUG // Variables below are only required for asserts; check for NDEBUG to avoid warnings.
  int n_dims = min.size();
  assert(n_dims==max.size());
  assert(n_dims==n_samples_per_dim.size());
#endif

  MatrixXd inputs;
  FunctionApproximator::generateInputsGrid(min, max, n_samples_per_dim, inputs);

  MatrixXd lines;
  getLines(inputs, lines);
  
  MatrixXd activations;
  if (cosine_basis_functions_)
  {
    BasisFunction::Cosine::activations(covars_,centers_,inputs,activations);
  }
  else
  {
    BasisFunction::Gaussian::activations(centers_,covars_,priors_,inputs,activations,normalized_basis_functions_);
    if (normalized_basis_functions_)
    {
      MatrixXd unnormalized_activations;
      BasisFunction::Gaussian::activations(centers_,covars_,priors_,inputs,unnormalized_activations,false);
      saveMatrix(save_directory,"activations_unnormalized_grid.txt",unnormalized_activations,overwrite);
    }
  }
    
  MatrixXd predictions;
  evaluate(inputs,predictions);
  
  saveMatrix(save_directory,"n_samples_per_dim.txt",n_samples_per_dim,overwrite);
  saveMatrix(save_directory,"inputs_grid.txt",inputs,overwrite);
  saveMatrix(save_directory,"lines_grid.txt",lines,overwrite);
  saveMatrix(save_directory,"activations_grid.txt",activations,overwrite);
  saveMatrix(save_directory,"predictions_grid.txt",predictions,overwrite);
  
  return true;
  
}

void UnifiedModel::setParameterVectorModifierPrivate(std::string modifier, bool new_value)
{
  if (modifier.compare("lines_pivot_at_max_activation")==0)
    set_lines_pivot_at_max_activation(new_value);
  
  if (modifier.compare("slopes_as_angles")==0)
    set_slopes_as_angles(new_value);
  
}

}
