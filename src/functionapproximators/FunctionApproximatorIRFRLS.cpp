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

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::FunctionApproximatorIRFRLS);

#include "functionapproximators/BasisFunction.hpp"
#include "functionapproximators/MetaParametersIRFRLS.hpp"
#include "functionapproximators/ModelParametersIRFRLS.hpp"

#include "dmpbbo_io/EigenBoostSerialization.hpp"
#include "dmpbbo_io/EigenFileIO.hpp"

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

FunctionApproximatorIRFRLS::FunctionApproximatorIRFRLS(const MetaParametersIRFRLS *const meta_parameters, const ModelParametersIRFRLS *const model_parameters) 
:
  FunctionApproximator(meta_parameters,model_parameters)
{
}

FunctionApproximatorIRFRLS::FunctionApproximatorIRFRLS(const ModelParametersIRFRLS *const model_parameters) 
:
  FunctionApproximator(model_parameters)
{
}


FunctionApproximator* FunctionApproximatorIRFRLS::clone(void) const {
  // All error checking and cloning is left to the FunctionApproximator constructor.
  return new FunctionApproximatorIRFRLS(
    dynamic_cast<const MetaParametersIRFRLS*>(getMetaParameters()),
    dynamic_cast<const ModelParametersIRFRLS*>(getModelParameters())
    );
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
  BasisFunction::Cosine::activations(cosines_periodes,cosines_phase,inputs,proj_inputs);
  
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
  
  const ModelParametersIRFRLS* model = static_cast<const ModelParametersIRFRLS*>(getModelParameters());

  MatrixXd proj_inputs;
  model->cosineActivations(input,proj_inputs);
  
  output = proj_inputs * model->weights_;
}

bool FunctionApproximatorIRFRLS::saveGridData(const VectorXd& min, const VectorXd& max, const VectorXi& n_samples_per_dim, string save_directory, bool overwrite) const
{
  if (save_directory.empty())
    return true;
  
  MatrixXd inputs_grid;
  FunctionApproximator::generateInputsGrid(min, max, n_samples_per_dim, inputs_grid);
      
  const ModelParametersIRFRLS* model_parameters_irfrls = static_cast<const ModelParametersIRFRLS*>(getModelParameters());
  
  MatrixXd activations_grid;
  model_parameters_irfrls->cosineActivations(inputs_grid, activations_grid);
  
  saveMatrix(save_directory,"n_samples_per_dim.txt",n_samples_per_dim,overwrite);
  saveMatrix(save_directory,"inputs_grid.txt",inputs_grid,overwrite);
  saveMatrix(save_directory,"activations_grid.txt",activations_grid,overwrite);

  // Weight the basis function activations  
  VectorXd weights = model_parameters_irfrls->weights();
  for (int b=0; b<activations_grid.cols(); b++)
    activations_grid.col(b).array() *= weights(b);
  saveMatrix(save_directory,"activations_weighted_grid.txt",activations_grid,overwrite);
  
  // Sum over weighed basis functions
  MatrixXd predictions_grid = activations_grid.rowwise().sum();
  saveMatrix(save_directory,"predictions_grid.txt",predictions_grid,overwrite);
  
  return true;
  
}

template<class Archive>
void FunctionApproximatorIRFRLS::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(FunctionApproximator);
}

}