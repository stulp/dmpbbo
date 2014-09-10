/**
 * @file   FunctionApproximatorLWPR.cpp
 * @brief  FunctionApproximator class source file.
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
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include "functionapproximators/FunctionApproximatorLWPR.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::FunctionApproximatorLWPR);

#include "dmpbbo_io/EigenBoostSerialization.hpp"

#include "functionapproximators/MetaParametersLWPR.hpp"
#include "functionapproximators/ModelParametersLWPR.hpp"

#include "lwpr.hh"

#include <iostream>

using namespace Eigen;
using namespace std;

namespace DmpBbo {

FunctionApproximatorLWPR::FunctionApproximatorLWPR(const MetaParametersLWPR *const meta_parameters, const ModelParametersLWPR *const model_parameters) 
:
  FunctionApproximator(meta_parameters,model_parameters),
  print_training_progress_(false)
{
}

FunctionApproximatorLWPR::FunctionApproximatorLWPR(const ModelParametersLWPR *const model_parameters) 
:
  FunctionApproximator(model_parameters),
  print_training_progress_(false)
{
}

FunctionApproximator* FunctionApproximatorLWPR::clone(void) const {
  // All error checking and cloning is left to the FunctionApproximator constructor.
  FunctionApproximatorLWPR* fa_lwpr = new FunctionApproximatorLWPR(
    dynamic_cast<const MetaParametersLWPR*>(getMetaParameters()),
    dynamic_cast<const ModelParametersLWPR*>(getModelParameters())
    );
  fa_lwpr->set_print_training_progress(print_training_progress_);
  return fa_lwpr;
};

void FunctionApproximatorLWPR::train(const MatrixXd& inputs, const MatrixXd& targets)
{
  if (isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorLWPR::train more than once. Doing nothing." << endl;
    cerr << "   (if you really want to retrain, call reTrain function instead)" << endl;
    return;
  }
  
  assert(inputs.rows() == targets.rows()); // Must have same number of examples
  assert(inputs.cols()==getExpectedInputDim());
  
  const MetaParametersLWPR* meta_parameters_lwpr = dynamic_cast<const MetaParametersLWPR*>(getMetaParameters());
  
  int n_in =inputs.cols();
  int n_out=targets.cols();
 
  LWPR_Object* lwpr_object = new LWPR_Object(n_in, n_out);
	lwpr_object->setInitD(    meta_parameters_lwpr->init_D_[0]); // todo Fix this
	lwpr_object->wGen(        meta_parameters_lwpr->w_gen_);
  lwpr_object->wPrune(      meta_parameters_lwpr->w_prune_);
  lwpr_object->updateD(     meta_parameters_lwpr->update_D_);
  lwpr_object->setInitAlpha(meta_parameters_lwpr->init_alpha_);
  lwpr_object->penalty(     meta_parameters_lwpr->penalty_);
	lwpr_object->diagOnly(    meta_parameters_lwpr->diag_only_);
  lwpr_object->useMeta(     meta_parameters_lwpr->use_meta_);
  lwpr_object->metaRate(    meta_parameters_lwpr->meta_rate_);
  lwpr_object->kernel(      meta_parameters_lwpr->kernel_name_.c_str());
   
  
  vector<double> input_vector(n_in);
  vector<double> target_vector(n_out);
  int n_input_samples = inputs.rows();
  
  //http://stackoverflow.com/questions/15858569/randomly-permute-rows-columns-of-a-matrix-with-eigen
  PermutationMatrix<Dynamic,Dynamic> permute(n_input_samples);	
  permute.setIdentity();
  VectorXi shuffled_indices = VectorXi::LinSpaced(n_input_samples,0,n_input_samples-1);
  MatrixXd outputs;
  for (int iterations=0; iterations<50; iterations++)
  {
    random_shuffle(permute.indices().data(), permute.indices().data()+permute.indices().size());
    shuffled_indices = (shuffled_indices.transpose()*permute).transpose();

    for (int ii=0; ii<n_input_samples; ii++)
    {
      // Eigen::VectorXd -> std::vector
      Map<VectorXd>(input_vector.data(),n_in) = inputs.row(shuffled_indices[ii]);  
      Map<VectorXd>(target_vector.data(),n_out) = targets.row(shuffled_indices[ii]);
      // update model
      lwpr_object->update(input_vector, target_vector);
    }

    if (print_training_progress_)
    {
      if (iterations%5==0)
      {
        FunctionApproximatorLWPR* fa_tmp = new FunctionApproximatorLWPR(new ModelParametersLWPR(new LWPR_Object(*lwpr_object)));
        fa_tmp->predict(inputs, outputs);
        delete fa_tmp;
        MatrixXd abs_error = (targets.array()-outputs.array()).abs();
        VectorXd mean_abs_error_per_output_dim = abs_error.colwise().mean();
        cout << "Iteration " << iterations << " MEA=" << mean_abs_error_per_output_dim << endl;
      }
    }
    
  }
  setModelParameters(new ModelParametersLWPR(lwpr_object));
  
}


void FunctionApproximatorLWPR::predict(const MatrixXd& inputs, MatrixXd& outputs)
{
  if (!isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorLWPR::predict if you have not trained yet. Doing nothing." << endl;
    return;
  }

  const ModelParametersLWPR* model_parameters_lwpr = static_cast<const ModelParametersLWPR*>(getModelParameters());

  int n_in  = model_parameters_lwpr->lwpr_object_->model.nIn;
  assert(inputs.cols()==n_in);
  
  int n_input_samples = inputs.rows();
  int n_out = model_parameters_lwpr->lwpr_object_->model.nOut;

  outputs.resize(n_input_samples,n_out);
  
  // Allocate memory for the temporary vectors for LWPR_Object::predict
  vector<double> input_vector(n_in);
  vector<double> output_vector(n_out);

  // Do prediction for each sample  
  for (int ii=0; ii<n_input_samples; ii++)
  {
    // LWPR_Object::predict uses std::vector, so do some conversions here.
    Map<VectorXd>(input_vector.data(),n_in) = inputs.row(ii);  // Eigen::VectorXd -> std::vector
    output_vector = model_parameters_lwpr->lwpr_object_->predict(input_vector);
    outputs.row(ii) = Map<VectorXd>(&output_vector[0], n_out); // std::vector -> Eigen::VectorXd
  }
  
}

template<class Archive>
void FunctionApproximatorLWPR::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(FunctionApproximator);
}


}
