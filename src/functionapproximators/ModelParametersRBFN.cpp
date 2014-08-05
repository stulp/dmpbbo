/**
 * @file   ModelParametersRBFN.cpp
 * @brief  ModelParametersRBFN class source file.
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
#include "functionapproximators/FunctionApproximator.hpp"
#include "functionapproximators/ModelParametersRBFN.hpp"

BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::ModelParametersRBFN);

#include "dmpbbo_io/EigenFileIO.hpp"
#include "dmpbbo_io/BoostSerializationToString.hpp"
#include "dmpbbo_io/EigenBoostSerialization.hpp"
#include "functionapproximators/ModelParametersUnified.hpp"

#include <iostream>
#include <fstream>

#include <eigen3/Eigen/Core>


using namespace std;
using namespace Eigen;

namespace DmpBbo {

ModelParametersRBFN::ModelParametersRBFN(const MatrixXd& centers, const MatrixXd& widths, const MatrixXd& weights) 
:
  centers_(centers),
  widths_(widths),
  weights_(weights),
  caching_(true)
{
#ifndef NDEBUG // Variables below are only required for asserts; check for NDEBUG to avoid warnings.
  int n_basis_functions = centers.rows();
  int n_dims = centers.cols();
#endif  
  assert(n_basis_functions==widths_.rows());
  assert(n_dims           ==widths_.cols());
  assert(n_basis_functions==weights_.rows());
  assert(1                ==weights_.cols());
  
  all_values_vector_size_ = 0;
  all_values_vector_size_ += centers_.rows()*centers_.cols();
  all_values_vector_size_ += widths_.rows() *widths_.cols();
  all_values_vector_size_ += weights_.rows()*weights_.cols();

};

ModelParameters* ModelParametersRBFN::clone(void) const {
  return new ModelParametersRBFN(centers_,widths_,weights_); 
}

void ModelParametersRBFN::kernelActivations(const MatrixXd& inputs, MatrixXd& kernel_activations) const
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
  kernelActivations(centers_,widths_,inputs,kernel_activations);

  if (caching_)
  {
    // Cache the current results now.  
    inputs_cached_ = inputs;
    kernel_activations_cached_ = kernel_activations;
  }
  
}

void ModelParametersRBFN::weightedBasisFunctions(const MatrixXd& inputs, MatrixXd& output) const
{  
  output.resize(inputs.rows(),1); // Fix this
  // Assert that memory has been pre-allocated.
  assert(inputs.rows()==output.rows());
  
  // Get the basis function activations  
  MatrixXd activations; // todo avoid allocation
  kernelActivations(inputs,activations);
    
  // Weight the basis function activations  
  for (int b=0; b<activations.cols(); b++)
    activations.col(b).array() *= weights_(b);

  // Sum over weighed basis functions
  output = activations.rowwise().sum();
    
}

void ModelParametersRBFN::kernelActivations(const MatrixXd& centers, const MatrixXd& widths, const MatrixXd& inputs, MatrixXd& kernel_activations)
{
  bool asymmetric_kernels=false;
  
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

  double c,w,x;
  for (int bb=0; bb<n_basis_functions; bb++)
  {

    // Here, we compute the values of a (unnormalized) multi-variate Gaussian:
    //   activation = exp(-0.5*(x-mu)*Sigma^-1*(x-mu))
    // Because Sigma is diagonal in our case, this simplifies to
    //   activation = exp(\sum_d=1^D [-0.5*(x_d-mu_d)^2/Sigma_(d,d)]) 
    //              = \prod_d=1^D exp(-0.5*(x_d-mu_d)^2/Sigma_(d,d)) 
    // This last product is what we compute below incrementally
    
    kernel_activations.col(bb).fill(1.0);
    for (int i_dim=0; i_dim<n_dims; i_dim++)
    {
      c = centers(bb,i_dim);
      for (int i_s=0; i_s<n_samples; i_s++)
      {
        x = inputs(i_s,i_dim);
        w = widths(bb,i_dim);
        
        if (asymmetric_kernels && x<c && bb>0)
          // Get the width of the previous basis function
          // This is the part that makes it assymetric
          w = widths(bb-1,i_dim);
          
        kernel_activations(i_s,bb) *= exp(-0.5*pow(x-c,2)/(w*w));
      }
    }
  }
}

template<class Archive>
void ModelParametersRBFN::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ModelParameters);

  ar & BOOST_SERIALIZATION_NVP(centers_);
  ar & BOOST_SERIALIZATION_NVP(widths_);
  ar & BOOST_SERIALIZATION_NVP(weights_);
  ar & BOOST_SERIALIZATION_NVP(all_values_vector_size_);
  ar & BOOST_SERIALIZATION_NVP(caching_);
}

string ModelParametersRBFN::toString(void) const
{
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("ModelParametersRBFN");
}

void ModelParametersRBFN::getSelectableParameters(set<string>& selected_values_labels) const 
{
  selected_values_labels = set<string>();
  selected_values_labels.insert("centers");
  selected_values_labels.insert("widths");
  selected_values_labels.insert("weights");
}


void ModelParametersRBFN::getParameterVectorMask(const std::set<std::string> selected_values_labels, VectorXi& selected_mask) const
{

  selected_mask.resize(getParameterVectorAllSize());
  selected_mask.fill(0);
  
  int offset = 0;
  int size;
  
  // Centers
  size = centers_.rows()*centers_.cols();
  if (selected_values_labels.find("centers")!=selected_values_labels.end())
    selected_mask.segment(offset,size).fill(1);
  offset += size;
  
  // Widths
  size = widths_.rows()*widths_.cols();
  if (selected_values_labels.find("widths")!=selected_values_labels.end())
    selected_mask.segment(offset,size).fill(2);
  offset += size;
  
  // Offsets
  size = weights_.rows()*weights_.cols();
  if (selected_values_labels.find("weights")!=selected_values_labels.end())
    selected_mask.segment(offset,size).fill(3);
  offset += size;

  offset += size;

  assert(offset == getParameterVectorAllSize());   
}

void ModelParametersRBFN::getParameterVectorAll(VectorXd& values) const
{
  values.resize(getParameterVectorAllSize());
  int offset = 0;
  
  for (int i_dim=0; i_dim<centers_.cols(); i_dim++)
  {
    values.segment(offset,centers_.rows()) = centers_.col(i_dim);
    offset += centers_.rows();
  }
  
  for (int i_dim=0; i_dim<widths_.cols(); i_dim++)
  {
    values.segment(offset,widths_.rows()) = widths_.col(i_dim);
    offset += widths_.rows();
  }
  
  values.segment(offset,weights_.rows()) = weights_;
  offset += weights_.rows();
  
  assert(offset == getParameterVectorAllSize());   
};

void ModelParametersRBFN::setParameterVectorAll(const VectorXd& values) {

  if (all_values_vector_size_ != values.size())
  {
    cerr << __FILE__ << ":" << __LINE__ << ": values is of wrong size." << endl;
    return;
  }
  
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
    if ( !(widths_.col(i_dim).array() == values.segment(offset,size).array()).all() )
      clearCache();
    
    widths_.col(i_dim) = values.segment(offset,size);
    offset += size;
  }

  weights_ = values.segment(offset,size);
  offset += size;
  // Cache must not be cleared, because kernelActivations() returns the same values.

  assert(offset == getParameterVectorAllSize());   
};

bool ModelParametersRBFN::saveGridData(const VectorXd& min, const VectorXd& max, const VectorXi& n_samples_per_dim, string save_directory, bool overwrite) const
{
  if (save_directory.empty())
    return true;
  
  MatrixXd inputs;
  FunctionApproximator::generateInputsGrid(min, max, n_samples_per_dim, inputs);
      
  MatrixXd activations;
  kernelActivations(inputs, activations);
    
  saveMatrix(save_directory,"n_samples_per_dim.txt",n_samples_per_dim,overwrite);
  saveMatrix(save_directory,"inputs_grid.txt",inputs,overwrite);
  saveMatrix(save_directory,"activations.txt",activations,overwrite);

  // Weight the basis function activations  
  for (int b=0; b<activations.cols(); b++)
    activations.col(b).array() *= weights_(b);
  saveMatrix(save_directory,"activations_weighted.txt",activations,overwrite);
  
  return true;
  
}

void ModelParametersRBFN::setParameterVectorModifierPrivate(std::string modifier, bool new_value)
{
}

ModelParametersUnified* ModelParametersRBFN::toModelParametersUnified(void) const
{
  cout << "ModelParametersRBFN::toModelParametersUnified" << endl;
  // RBFN uses degenerate line models, i.e. with zero slopes
  MatrixXd slopes = MatrixXd::Zero(centers_.rows(),centers_.cols());
  // RBFN does not use normalized basis functions
  bool normalized_basis_functions = false;
  return new ModelParametersUnified(centers_, widths_, slopes, weights_,normalized_basis_functions); 
  
}

}


