/**
 * @file   DistributionGaussian.cpp
 * @brief  DistributionGaussian class source file.
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

#include "bbo/DistributionGaussian.hpp"

#include <iomanip>

#include <boost/random.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigenvalues>

#include "dmpbbo_io/EigenBoostSerialization.hpp"


using namespace std;
using namespace Eigen;

namespace DmpBbo {

// Initialize random number generator
boost::mt19937 DistributionGaussian::rng = boost::mt19937(getpid() + time(0));

// Initialize uni-variate unit normal distribution
boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >
*DistributionGaussian::normal_distribution_unit
   = new boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >(
        rng, 
        boost::normal_distribution<>(0, 1)
     );


DistributionGaussian::DistributionGaussian(const VectorXd& mean, const MatrixXd& covar) 
{
  mean_ = mean;
  set_covar(covar);
}

DistributionGaussian* DistributionGaussian::clone(void) const
{
  return new DistributionGaussian(mean(),covar()); 
}

void DistributionGaussian::set_mean(const VectorXd& mean) 
{ 
  assert(mean.size()==mean_.size());  
  mean_ = mean;
}

double DistributionGaussian::maxEigenValue(void) const
{
  if (max_eigen_value_<0.0)
  {
    SelfAdjointEigenSolver<MatrixXd> eigensolver(covar_);
    if (eigensolver.info() == Success)
    {
      // Get the eigenvalues
      VectorXd eigen_values = eigensolver.eigenvalues();
      max_eigen_value_ = eigen_values.maxCoeff();
    }
  }
  return max_eigen_value_;
}



void DistributionGaussian::set_covar(const MatrixXd& covar) { 
  assert(covar.cols()==covar.rows());
  assert(covar.rows()==mean_.size());
  covar_ = covar;
  covar_decomposed_ = MatrixXd::Zero(0,0);
  max_eigen_value_ = -1.0;
}

void DistributionGaussian::generateSamples(int n_samples, MatrixXd& samples) const
{
  if (covar_decomposed_.size()==0)
  {
    // Now perform the Cholesky decomposition, which makes it easier to generate samples. 
    MatrixXd A(covar_.llt().matrixL());
    covar_decomposed_ = A;
    // Remark: it would have been better to do this in the constructor and set_covar, 
    // but I couldn't get it to work with boost::serialization (I tried hard)
  }
  
  
  int n_dims = mean_.size();
  samples.resize(n_samples,n_dims);
  
  // http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
  VectorXd z(n_dims);
  for (int i_sample=0; i_sample<n_samples; i_sample++)
  {
    // Generate vector with samples from standard normal distribution N(0,1) 
    for (int i_dim=0; i_dim<n_dims; i_dim++)
      z(i_dim) = (*normal_distribution_unit)();

    // Compute x = mu + Az
    samples.row(i_sample) = mean_ + covar_decomposed_*z;
  }  
}


std::ostream& operator<<(std::ostream& output, const DistributionGaussian& distribution)
{
  output << "N([" << toString(distribution.mean_) << "], ["<< toString(distribution.covar_) << "])";
  return output;
}


}
