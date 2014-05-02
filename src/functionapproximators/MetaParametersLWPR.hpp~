/**
 * @file   MetaParametersLWPR.hpp
 * @brief  MetaParametersLWPR class header file.
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

#ifndef METAPARAMETERSLWPR_H
#define METAPARAMETERSLWPR_H

#include "functionapproximators/MetaParameters.hpp"

#include <eigen3/Eigen/Core>

namespace DmpBbo {

/** \brief Meta-parameters for the Locally Weighted Projection Regression (LWPR) function approximator
 * \ingroup FunctionApproximators
 * \ingroup LWPR
 */
class MetaParametersLWPR : public MetaParameters
{
  friend class FunctionApproximatorLWPR;

public:
  
  /** Constructor for the algorithmic meta-parameters of the LWPR function approximator.
   *
   *  The meaning of these parameters is explained here:
   *     http://wcms.inf.ed.ac.uk/ipab/slmc/research/software-lwpr
   *
   *  Short howto:
   *  \li Set update_D=false, diag_only=true, use_meta=false
   *  \li Tune init_D, and then w_gen and w_prune
   *  \li Set update_D=true, then tune init_alpha, then penalty
   *  \li Set diag_only=false, see if it helps, if so re-tune init_alpha if necessary
   *  \li Set use_meta=true, tune meta_rate (I never do this...)  
   *
   *  \param[in] expected_input_dim Expected dimensionality of the input data
   *
   *  \param[in] init_D  Removing/adding kernels: Width of a kernel when it is newly placed. Smaller *                     values mean wider kernels.
   *  \param[in] w_gen   Removing/adding kernels: Threshold for adding a kernel.
   *  \param[in] w_prune Removing/adding kernels: Threshold for pruning a kernel.
   *
   *  \param[in] update_D   Updating existing kernels: whether to update kernels
   *  \param[in] init_alpha Updating existing kernels: rate at which kernels are updated
   *  \param[in] penalty    Updating existing kernels: regularization term. Higher penalty means
   *                        less kernels.
   *  \param[in] diag_only  Whether to update only the diagonal of the covariance matrix of the
   *                        kernel, or the full matrix. 
   * 
   *  \param[in] use_meta    Meta-learning of kernel update rate: whether meta-learning is enabled
   *  \param[in] meta_rate   Meta-learning of kernel update rate: meta-learning rate
   * 
   *  \param[in] kernel_name Removing/adding kernels: Type of kernels
   */
	MetaParametersLWPR(
	  int expected_input_dim,
    Eigen::VectorXd init_D=Eigen::VectorXd::Ones(1),
    double   w_gen=0.2,
    double   w_prune=0.8,
             
    bool     update_D=true,
    double   init_alpha=1.0,
    double   penalty=1.0,
    bool     diag_only=true,
             
    bool     use_meta=false,
    double   meta_rate=1.0,
    std::string   kernel_name=std::string("Gaussian")
  );
		
	MetaParametersLWPR* clone(void) const;

  std::string toString(void) const;

private:
  Eigen::VectorXd init_D_; // If of size 1, call setInitD(double), else setInitD(vector<double>)
  double w_gen_;
  double w_prune_;
    
  bool   update_D_;
  double init_alpha_;
  double penalty_;
  bool   diag_only_;
  
  bool   use_meta_;
  double meta_rate_;
  std::string kernel_name_; // "Gaussian" "BiSquare"

  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
   MetaParametersLWPR(void) {};
   
  /** Give boost serialization access to private members. */  
  friend class boost::serialization::access;
  
  /** Serialize class data members to boost archive. 
   * \param[in] ar Boost archive
   * \param[in] version Version of the class
   * See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/tutorial.html#simplecase
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version);

};

}

#include <boost/serialization/export.hpp>
/** Register this derived class. */
BOOST_CLASS_EXPORT_KEY2(DmpBbo::MetaParametersLWPR, "MetaParametersLWPR")

/** Don't add version information to archives. */
BOOST_CLASS_IMPLEMENTATION(DmpBbo::MetaParametersLWPR,boost::serialization::object_serializable);

#endif        //  #ifndef METAPARAMETERSLWPR_H

