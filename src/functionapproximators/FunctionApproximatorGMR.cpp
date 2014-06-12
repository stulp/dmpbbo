/**
 * @file   FunctionApproximatorGMR.cpp
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
#include "functionapproximators/FunctionApproximatorGMR.hpp"

BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::FunctionApproximatorGMR);

#include <iostream> 
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Cholesky>
#include <ctime>
#include <cstdlib>


#include "functionapproximators/ModelParametersGMR.hpp"
#include "functionapproximators/MetaParametersGMR.hpp"
#include "dmpbbo_io/EigenBoostSerialization.hpp"
#include "dmpbbo_io/EigenFileIO.hpp"


using namespace std;
using namespace Eigen;

namespace DmpBbo {

FunctionApproximatorGMR::FunctionApproximatorGMR(MetaParametersGMR* meta_parameters, ModelParametersGMR* model_parameters)
:
  FunctionApproximator(meta_parameters,model_parameters)
{
  // TODO : find a more appropriate place for rand initialization
  srand(unsigned(time(0)));
}

FunctionApproximatorGMR::FunctionApproximatorGMR(ModelParametersGMR* model_parameters)
:
  FunctionApproximator(model_parameters)
{
}

FunctionApproximator* FunctionApproximatorGMR::clone(void) const {
  MetaParametersGMR* meta_params  = NULL;
  if (getMetaParameters()!=NULL)
    meta_params = dynamic_cast<MetaParametersGMR*>(getMetaParameters()->clone());

  ModelParametersGMR* model_params = NULL;
  if (getModelParameters()!=NULL)
    model_params = dynamic_cast<ModelParametersGMR*>(getModelParameters()->clone());

  if (meta_params==NULL)
    return new FunctionApproximatorGMR(model_params);
  else
    return new FunctionApproximatorGMR(meta_params,model_params);
};

void FunctionApproximatorGMR::train(const MatrixXd& inputs, const MatrixXd& targets)
{
  if (isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorGMR::train more than once. Doing nothing." << endl;
    cerr << "   (if you really want to retrain, call reTrain function instead)" << endl;
    return;
  }
  
  assert(inputs.rows() == targets.rows()); // Must have same number of examples
  assert(inputs.cols()==getExpectedInputDim());

  const MetaParametersGMR* meta_parameters_GMR = 
    static_cast<const MetaParametersGMR*>(getMetaParameters());

  int n_gaussians = meta_parameters_GMR->number_of_gaussians_;
  int n_dims_in = inputs.cols();
  int n_dims_out = targets.cols();
  int n_dims_gmm = n_dims_in + n_dims_out;
  
  // Initialize the means, priors and covars
  std::vector<VectorXd> means(n_gaussians);
  std::vector<MatrixXd> covars(n_gaussians);
  std::vector<double> priors(n_gaussians);
  for (int i = 0; i < n_gaussians; i++)
  {
    means[i] = VectorXd(n_dims_gmm);
    priors[i] = 0.0;
    covars[i] = MatrixXd(n_dims_gmm, n_dims_gmm);
  }
  
  // Put the input/output data in one big matrix
  MatrixXd data = MatrixXd(inputs.rows(), n_dims_gmm);
  data << inputs, targets;

  // Initialization
  if (inputs.cols() == 1)
    firstDimSlicingInit(data, means, priors, covars);
  else
    kMeansInit(data, means, priors, covars);
  
  // Expectation-Maximization
  expectationMaximization(data, means, priors, covars);

  // Extract the different input/output components from the means/covars which contain both
  std::vector<Eigen::VectorXd> mean_xs(n_gaussians);
  std::vector<Eigen::VectorXd> mean_ys(n_gaussians);
  std::vector<Eigen::MatrixXd> covar_xs(n_gaussians);
  std::vector<Eigen::MatrixXd> covar_ys(n_gaussians);
  std::vector<Eigen::MatrixXd> covar_y_xs(n_gaussians);
  for (int i_gaussian = 0; i_gaussian < n_gaussians; i_gaussian++)
  {
    mean_xs[i_gaussian]    = means[i_gaussian].segment(0, n_dims_in);
    mean_ys[i_gaussian]    = means[i_gaussian].segment(n_dims_in, n_dims_out);

    covar_xs[i_gaussian]   = covars[i_gaussian].block(0, 0, n_dims_in, n_dims_in);
    covar_ys[i_gaussian]   = covars[i_gaussian].block(n_dims_in, n_dims_in, n_dims_out, n_dims_out);
    covar_y_xs[i_gaussian] = covars[i_gaussian].block(n_dims_in, 0, n_dims_out, n_dims_in);
  }


  setModelParameters(new ModelParametersGMR(priors, mean_xs, mean_ys, covar_xs, covar_ys, covar_y_xs));

  // std::vector<VectorXd> centers;
  // std::vector<MatrixXd> slopes;
  // std::vector<VectorXd> biases;
  // std::vector<MatrixXd> inverseCovarsL;

  // // int n_dims_in = inputs.cols();
  // // int n_dims_out = targets.cols();

  // for (int i_gaussian = 0; i_gaussian < n_gaussians; i_gaussian++)
  // {
  //   centers.push_back(VectorXd(means[i_gaussian].segment(0, n_dims_in)));

  //   slopes.push_back(MatrixXd(covars[i_gaussian].block(n_dims_in, 0, n_dims_out, n_dims_in) * covars[i_gaussian].block(0, 0, n_dims_in, n_dims_in).inverse()));
    
  //   biases.push_back(VectorXd(means[i_gaussian].segment(n_dims_in, n_dims_out) -
  //     slopes[i_gaussian]*means[i_gaussian].segment(0, n_dims_in)));

  //   MatrixXd L = covars[i_gaussian].block(0, 0, n_dims_in, n_dims_in).inverse().llt().matrixL();
  //   inverseCovarsL.push_back(MatrixXd(L));
  // }

  // setModelParameters(new ModelParametersGMR(centers, priors, slopes, biases, inverseCovarsL));

  //for (size_t i = 0; i < means.size(); i++)
  //  delete means[i];
  //for (size_t i = 0; i < covars.size(); i++)
  //delete covars[i];
}

double FunctionApproximatorGMR::normalPDF(const VectorXd& mu, const MatrixXd& covar, const VectorXd& input)
{
  VectorXd diff = input-mu;
  double output = exp(-2*diff.transpose()*covar.inverse()*diff);
  output *= pow(pow(2*M_PI,mu.size())*covar.determinant(),-0.5);   //  ( (2\pi)^N*|\Sigma| )^(-1/2)
  
  return output;
}

void FunctionApproximatorGMR::predict(const MatrixXd& inputs, MatrixXd& outputs)
{
  if (!isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorGMR::predict if you have not trained yet. Doing nothing." << endl;
    return;
  }
  
  const ModelParametersGMR* gmm = static_cast<const ModelParametersGMR*>(getModelParameters());

  // Dimensionality of input must be same as of the gmm inputs  
  assert(gmm->means_x_[0].size()==inputs.cols());
  
  int n_gaussians = gmm->priors_.size();
  assert(n_gaussians>0);
  int n_dims_out = gmm->means_y_[0].size();
  int n_inputs = inputs.rows();


  // Make outputs of the right size
  outputs.resize(n_inputs, n_dims_out);
  outputs.fill(0);
  
  // Pre-allocate some memory
  VectorXd prior_times_gauss(n_gaussians);
  VectorXd h(n_gaussians);
  for (int i_input=0; i_input<n_inputs; i_input++)
  {
    // Compute output for this input
    VectorXd input = inputs.row(i_input);
    
    // Compute gaussian pdf and multiply it with prior probability
    for (int i_gau=0; i_gau<n_gaussians; i_gau++)
    {
      double gauss = normalPDF(gmm->means_x_[i_gau],gmm->covars_x_[i_gau],input);
      prior_times_gauss(i_gau) = gmm->priors_[i_gau]*gauss;
    }

    // Normalize h
    VectorXd h = prior_times_gauss/prior_times_gauss.sum();
    
    // Compute output, given probabilities of each Gaussian
    for (int i_gau=0; i_gau<n_gaussians; i_gau++)
    {
      VectorXd diff = input-gmm->means_x_[i_gau];
      VectorXd projected =  gmm->covars_y_x_[i_gau] * gmm->covars_x_[i_gau].inverse() * diff;
      outputs.row(i_input) += h[i_gau] * (gmm->means_y_[i_gau]+projected);
    }
  }
}

void FunctionApproximatorGMR::firstDimSlicingInit(const MatrixXd& data, std::vector<VectorXd>& centers, std::vector<double>& priors,
  std::vector<MatrixXd>& covars)
{

  VectorXd first_dim = data.col(0);

  VectorXi assign(data.rows());
  assign.setZero();

  double min_val = first_dim.minCoeff();
  double max_val = first_dim.maxCoeff();

  for (int i_first_dim = 0; i_first_dim < first_dim.size(); i_first_dim++)
  {
    unsigned int center = int((first_dim[i_first_dim]-min_val)/(max_val-min_val)*centers.size());
    if (center==centers.size())
      center--;
    assign[i_first_dim] = center;
  }
  
  // Init means
  VectorXi nbPoints = VectorXi::Zero(centers.size());
  for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
    centers[i_gaussian].setZero();
  for (int iData = 0; iData < data.rows(); iData++)
  {
    centers[assign[iData]] += data.row(iData).transpose();
    nbPoints[assign[iData]]++;
  }
  for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
    centers[i_gaussian] /= nbPoints[i_gaussian];

  // Init covars
  for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
    covars[i_gaussian].setZero();
  for (int iData = 0; iData < data.rows(); iData++)
    covars[assign[iData]] += (data.row(iData).transpose() - centers[assign[iData]]) * (data.row(iData).transpose() - centers[assign[iData]]).transpose();
  for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
    covars[i_gaussian] /= nbPoints[i_gaussian];

  // Be sure that covar is invertible
  for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
      covars[i_gaussian] += MatrixXd::Identity(covars[i_gaussian].rows(), covars[i_gaussian].cols()) * 1e-5;

  // Init priors
  for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
    priors[i_gaussian] = 1. / centers.size();
}

void FunctionApproximatorGMR::kMeansInit(const MatrixXd& data, std::vector<VectorXd>& centers, std::vector<double>& priors,
  std::vector<MatrixXd>& covars, int n_max_iter)
{

  MatrixXd dataCentered = data.rowwise() - data.colwise().mean();
  MatrixXd dataCov = dataCentered.transpose() * dataCentered / data.rows();
  MatrixXd dataCovInverse = dataCov.inverse();

  std::vector<int> dataIndex;
  for (int i = 0; i < data.rows(); i++)
    dataIndex.push_back(i); 
  std::random_shuffle (dataIndex.begin(), dataIndex.end());

  for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
    centers[i_gaussian] = data.row(dataIndex[i_gaussian]);

  VectorXi assign(data.rows());
  assign.setZero();

  bool converged = false;
  for (int iIter = 0; iIter < n_max_iter && !converged; iIter++)
  {
    // E step
    converged = true;
    for (int iData = 0; iData < data.rows(); iData++)
    {
      VectorXd v = (centers[assign[iData]] - data.row(iData).transpose());

      double minDist = v.transpose() * dataCovInverse * v;

      for (int i_gaussian = 0; i_gaussian < (int)centers.size(); i_gaussian++)
      {
        if (i_gaussian == assign[iData])
          continue;

        v = (centers[i_gaussian] - data.row(iData).transpose());
        double dist = v.transpose() * dataCovInverse * v;
        if (dist < minDist)
        {
          converged = false;
          minDist = dist;
          assign[iData] = i_gaussian;
        }
      }
    }

    // M step
    VectorXi nbPoints = VectorXi::Zero(centers.size());
    for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
      centers[i_gaussian].setZero();
    for (int iData = 0; iData < data.rows(); iData++)
    {
      centers[assign[iData]] += data.row(iData).transpose();
      nbPoints[assign[iData]]++;
    }
    for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
      centers[i_gaussian] /= nbPoints[i_gaussian];
  }

  // Init covars
  VectorXi nbPoints = VectorXi::Zero(centers.size());
  for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
    covars[i_gaussian].setZero();
  for (int iData = 0; iData < data.rows(); iData++)
  {
    covars[assign[iData]] += (data.row(iData).transpose() - centers[assign[iData]]) * (data.row(iData).transpose() - centers[assign[iData]]).transpose();
    nbPoints[assign[iData]]++;
  }
  for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
    covars[i_gaussian] /= nbPoints[i_gaussian];

  // Be sure that covar is invertible
  for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
    covars[i_gaussian] += MatrixXd::Identity(covars[i_gaussian].rows(), covars[i_gaussian].cols()) * 1e-5f;

  // Init priors
  for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
    priors[i_gaussian] = 1. / centers.size();
}

void saveGMM(string directory, const vector<VectorXd>& centers, const vector<MatrixXd>& covars, int iter)
{
  for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
  {
    stringstream stream;
    stream << "gmm_iter" << setw(2) << setfill('0') << iter << "_gauss" << i_gaussian << "_mu.txt";
    string filename = stream.str();
    if (!saveMatrix(directory, filename,  centers[i_gaussian],  true))
      exit(0);
    cout << "  filename=" << filename << endl;
    
    stringstream stream2;
    stream2 << "gmm_iter" << setw(2) << setfill('0') << iter << "_gauss" << i_gaussian << "_covar.txt";
    filename = stream2.str();
    cout << "  filename=" << filename << endl;
    
    if (!saveMatrix(directory, filename,  covars[i_gaussian],  true))
      exit(0);
    
  }
}

void FunctionApproximatorGMR::expectationMaximization(const MatrixXd& data, std::vector<VectorXd>& centers, std::vector<double>& priors,
    std::vector<MatrixXd>& covars, int n_max_iter)
{
  MatrixXd assign(centers.size(), data.rows());
  assign.setZero();

  double oldLoglik = -1e10f;
  double loglik = 0;

  for (int iIter = 0; iIter < n_max_iter; iIter++)
  {
    // For debugging only
    //saveGMM("/tmp/demoTrainFunctionApproximators/GMR",centers,covars,iIter);
    
    // E step
    for (int iData = 0; iData < data.rows(); iData++)
      for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
        assign(i_gaussian, iData) = priors[i_gaussian] * FunctionApproximatorGMR::normalPDF(centers[i_gaussian], covars[i_gaussian],data.row(iData).transpose());

    oldLoglik = loglik;
    loglik = 0;
    for (int iData = 0; iData < data.rows(); iData++)
      loglik += log(assign.col(iData).sum());
    loglik /= data.rows();

    if (fabs(loglik / oldLoglik - 1) < 1e-8f)
      break;

    for (int iData = 0; iData < data.rows(); iData++)
      assign.col(iData) /= assign.col(iData).sum();

    // M step
    for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
    {
      centers[i_gaussian].setZero();
      covars[i_gaussian].setZero();
      priors[i_gaussian] = 0;
    }

    for (int iData = 0; iData < data.rows(); iData++)
    {
      for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
      {
        centers[i_gaussian] += assign(i_gaussian, iData) * data.row(iData).transpose();
        priors[i_gaussian] += assign(i_gaussian, iData);
      }
    }

    for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
    {
      centers[i_gaussian] /= assign.row(i_gaussian).sum();
      priors[i_gaussian] /= assign.cols();
    }

    for (int iData = 0; iData < data.rows(); iData++)
      for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
        covars[i_gaussian] += assign(i_gaussian, iData) * (data.row(iData).transpose() - centers[i_gaussian]) * (data.row(iData).transpose() - centers[i_gaussian]).transpose();

    for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
      covars[i_gaussian] /= assign.row(i_gaussian).sum();

    // Be sure that covar is invertible
    for (size_t i_gaussian = 0; i_gaussian < centers.size(); i_gaussian++)
      covars[i_gaussian] += MatrixXd::Identity(covars[i_gaussian].rows(), covars[i_gaussian].cols()) * 1e-5f;
  }
}

template<class Archive>
void FunctionApproximatorGMR::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(FunctionApproximator);
}

}