/**
 * @file   FunctionApproximatorIRFRLS.cpp
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
#include "functionapproximators/FunctionApproximatorIRFRLS.hpp"

BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::FunctionApproximatorIRFRLS);

#include "functionapproximators/MetaParametersIRFRLS.hpp"
#include "functionapproximators/ModelParametersIRFRLS.hpp"

#include "dmpbbo_io/EigenBoostSerialization.hpp"

#include <eigen3/Eigen/LU>

#include <boost/math/constants/constants.hpp>
#include <boost/random.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real.hpp>

#include <iostream>

using namespace Eigen;
using namespace std;

namespace DmpBbo {

FunctionApproximatorIRFRLS::FunctionApproximatorIRFRLS(MetaParametersIRFRLS* meta_parameters, ModelParametersIRFRLS* model_parameters)
:
  FunctionApproximator(meta_parameters,model_parameters)
{
}

FunctionApproximatorIRFRLS::FunctionApproximatorIRFRLS(ModelParametersIRFRLS* model_parameters)
:
  FunctionApproximator(model_parameters)
{
}

FunctionApproximator* FunctionApproximatorIRFRLS::clone(void) const {
  MetaParametersIRFRLS*  meta_params  = NULL;
  if (getMetaParameters()!=NULL)
    meta_params = dynamic_cast<MetaParametersIRFRLS*>(getMetaParameters()->clone());

  ModelParametersIRFRLS* model_params = NULL;
  if (getModelParameters()!=NULL)
    model_params = dynamic_cast<ModelParametersIRFRLS*>(getModelParameters()->clone());

  if (meta_params==NULL)
    return new FunctionApproximatorIRFRLS(model_params);
  else
    return new FunctionApproximatorIRFRLS(meta_params,model_params);
};


void FunctionApproximatorIRFRLS::train(const MatrixXd& inputs, const MatrixXd& targets)
{
  if (isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorIRFRLS::train more than once. Doing nothing." << endl;
    cerr << "   (if you really want to retrain, call reTrain function instead)" << endl;
    return;
  }
  
  assert(inputs.rows() == targets.rows()); // Must have same number of examples
  assert(inputs.cols()==getExpectedInputDim());
  
  const MetaParametersIRFRLS* meta_parameters_irfrls = 
    static_cast<const MetaParametersIRFRLS*>(getMetaParameters());

  int nb_cos = meta_parameters_irfrls->number_of_basis_functions_;

  // Init random generator.
  boost::mt19937 rng(getpid() + time(0));

  // Draw periodes
  boost::normal_distribution<> twoGamma(0, sqrt(2 * meta_parameters_irfrls->gamma_));
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > genPeriods(rng, twoGamma);
  MatrixXd cosines_periodes(nb_cos, inputs.cols());
  for (int r = 0; r < nb_cos; r++)
    for (int c = 0; c < inputs.cols(); c++)
      cosines_periodes(r, c) = genPeriods();

  // Draw phase
  boost::uniform_real<> twoPi(0, 2 * boost::math::constants::pi<double>());
  boost::variate_generator<boost::mt19937&, boost::uniform_real<> > genPhases(rng, twoPi);
  VectorXd cosines_phase(nb_cos);
  for (int r = 0; r < nb_cos; r++)
      cosines_phase(r) = genPhases();

  MatrixXd proj_inputs;
  proj(inputs, cosines_periodes, cosines_phase, proj_inputs);

  // Compute linear model analatically
  double lambda = meta_parameters_irfrls->lambda_;
  MatrixXd toInverse = lambda * MatrixXd::Identity(nb_cos, nb_cos) + proj_inputs.transpose() * proj_inputs;
  VectorXd linear_model = toInverse.inverse() *
    (proj_inputs.transpose() * targets);

  setModelParameters(new ModelParametersIRFRLS(linear_model, cosines_periodes, cosines_phase));
}

void FunctionApproximatorIRFRLS::predict(const MatrixXd& input, MatrixXd& output)
{
  if (!isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorIRFRLS::predict if you have not trained yet. Doing nothing." << endl;
    return;
  }
  
  const ModelParametersIRFRLS* model_parameters_irfrls = static_cast<const ModelParametersIRFRLS*>(getModelParameters());

  MatrixXd proj_inputs;
  proj(input, model_parameters_irfrls->cosines_periodes_, model_parameters_irfrls->cosines_phase_, proj_inputs);
  output = proj_inputs * model_parameters_irfrls->linear_models_;
}

/** Cosinus function. Used to select correct overload version of cos in FunctionApproximatorIRFRLS::proj
 * \param[in]  x A decimal number
 * \return The cosinus of x
 */
double double_cosine(double x)
{
  return cos(x);
}

void FunctionApproximatorIRFRLS::proj(const MatrixXd& vecs, const MatrixXd& periods, const VectorXd& phases, Eigen::MatrixXd& projected)
{
  projected = vecs * periods.transpose();
  projected.rowwise() += phases.transpose();
  projected = projected.unaryExpr(ptr_fun(double_cosine));
}

template<class Archive>
void FunctionApproximatorIRFRLS::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(FunctionApproximator);
}

}