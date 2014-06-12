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

  int nbGaussian = meta_parameters_GMR->number_of_gaussians_;
  int gmmDim = inputs.cols() + targets.cols();

  std::vector<VectorXd> gmmCenters;
  std::vector<double> gmmPriors;
  std::vector<MatrixXd> gmmCovars;

  for (int i = 0; i < nbGaussian; i++)
  {
    gmmCenters.push_back(VectorXd(gmmDim));
    gmmPriors.push_back(0.0);
    gmmCovars.push_back(MatrixXd(gmmDim, gmmDim));
  }

  MatrixXd gmmData = MatrixXd(inputs.rows(), gmmDim);
  gmmData << inputs, targets;

  if (inputs.cols() == 1)
    firstDimSlicingInit(gmmData, gmmCenters, gmmPriors, gmmCovars);
  else
    kMeansInit(gmmData, gmmCenters, gmmPriors, gmmCovars);
  EM(gmmData, gmmCenters, gmmPriors, gmmCovars);

  std::vector<Eigen::VectorXd> mu_xs;
  std::vector<Eigen::VectorXd> mu_ys;

  std::vector<Eigen::MatrixXd> sigma_xs;
  std::vector<Eigen::MatrixXd> sigma_ys;
  std::vector<Eigen::MatrixXd> sigma_y_xs;

  int nbInDim = inputs.cols();
  int nbOutDim = targets.cols();

  for (int iCenter = 0; iCenter < nbGaussian; iCenter++)
  {
    mu_xs.push_back(gmmCenters[iCenter].segment(0, nbInDim));
    mu_ys.push_back(gmmCenters[iCenter].segment(nbInDim, nbOutDim));

    sigma_xs.push_back(gmmCovars[iCenter].block(0, 0, nbInDim, nbInDim));
    sigma_ys.push_back(gmmCovars[iCenter].block(nbInDim, nbInDim, nbOutDim, nbOutDim));
    sigma_y_xs.push_back(gmmCovars[iCenter].block(nbInDim, 0, nbOutDim, nbInDim));
  }


  setModelParameters(new ModelParametersGMR(gmmPriors, mu_xs, mu_ys, sigma_xs, sigma_ys, sigma_y_xs));

  // std::vector<VectorXd> centers;
  // std::vector<MatrixXd> slopes;
  // std::vector<VectorXd> biases;
  // std::vector<MatrixXd> inverseCovarsL;

  // // int nbInDim = inputs.cols();
  // // int nbOutDim = targets.cols();

  // for (int iCenter = 0; iCenter < nbGaussian; iCenter++)
  // {
  //   centers.push_back(VectorXd(gmmCenters[iCenter].segment(0, nbInDim)));

  //   slopes.push_back(MatrixXd(gmmCovars[iCenter].block(nbInDim, 0, nbOutDim, nbInDim) * gmmCovars[iCenter].block(0, 0, nbInDim, nbInDim).inverse()));
    
  //   biases.push_back(VectorXd(gmmCenters[iCenter].segment(nbInDim, nbOutDim) -
  //     slopes[iCenter]*gmmCenters[iCenter].segment(0, nbInDim)));

  //   MatrixXd L = gmmCovars[iCenter].block(0, 0, nbInDim, nbInDim).inverse().llt().matrixL();
  //   inverseCovarsL.push_back(MatrixXd(L));
  // }

  // setModelParameters(new ModelParametersGMR(centers, gmmPriors, slopes, biases, inverseCovarsL));

  //for (size_t i = 0; i < gmmCenters.size(); i++)
  //  delete gmmCenters[i];
  //for (size_t i = 0; i < gmmCovars.size(); i++)
  //delete gmmCovars[i];
}

double gaussian(const VectorXd& mu, const MatrixXd& covar, const VectorXd input)
{
  VectorXd diff = input-mu;
  double output = exp(-2*diff.transpose()*covar.inverse()*diff);
  output *= pow(pow(2*M_PI,mu.size())*covar.determinant(),-0.5);   //  ( (2\pi)^N*|\Sigma| )^(-1/2)
  
  return output;
}

void FunctionApproximatorGMR::predict(const MatrixXd& input, MatrixXd& output)
{
  if (!isTrained())  
  {
    cerr << "WARNING: You may not call FunctionApproximatorGMR::predict if you have not trained yet. Doing nothing." << endl;
    return;
  }
  
  const ModelParametersGMR* model_parameters_GMR = static_cast<const ModelParametersGMR*>(getModelParameters());

  int nbGaussian = model_parameters_GMR->priors_.size();
  //int nbInDim = model_parameters_GMR->mu_xs_[0].size();
  int nbOutDim = model_parameters_GMR->mu_ys_[0].size();


  output.resize(input.rows(), nbOutDim);
  output.fill(0);
  for (int i = 0; i < input.rows(); i++)
  {
    // Compute output for this input
    VectorXd inputPoint = input.row(i);
    
    // Compute gaussian probability and multiply it with prior probability
    VectorXd gauss(nbGaussian);
    VectorXd prior_times_gauss(nbGaussian);
    for (int iCenter = 0; iCenter < nbGaussian; iCenter++)
    {
      VectorXd center = model_parameters_GMR->mu_xs_[iCenter];
      MatrixXd covar = model_parameters_GMR->sigma_xs_[iCenter];
      gauss(iCenter) = gaussian(center,covar,inputPoint);
      prior_times_gauss(iCenter) = model_parameters_GMR->priors_[iCenter]*gauss(iCenter);
      
    }

    // Normalize h
    VectorXd h = prior_times_gauss/prior_times_gauss.sum();
    
    // Compute output, given probabilities of each Gaussian
    for (int iCenter = 0; iCenter < nbGaussian; iCenter++)
    {
      VectorXd diff = inputPoint-model_parameters_GMR->mu_xs_[iCenter];
      VectorXd projected =  model_parameters_GMR->sigma_y_xs_[iCenter] * model_parameters_GMR->sigma_xs_[iCenter].inverse()*diff;
      output.row(i) += h[iCenter] * (model_parameters_GMR->mu_ys_[iCenter]+projected);
    }
  }
}

void FunctionApproximatorGMR::firstDimSlicingInit(const MatrixXd& data, std::vector<VectorXd>& centers, std::vector<double>& priors,
  std::vector<MatrixXd>& covars)
{

  VectorXd firstDim = data.col(0);

  VectorXi assign(data.rows());
  assign.setZero();

  double minVal = firstDim.minCoeff();
  double maxVal = firstDim.maxCoeff();

  for (int iFirstDim = 0; iFirstDim < firstDim.size(); iFirstDim++)
  {
    size_t center = int((firstDim[iFirstDim] - minVal) / (maxVal - minVal) * centers.size());

    if (center == centers.size())
      center--;

    assign[iFirstDim] = center;
  }
  
  // Init means
  VectorXi nbPoints = VectorXi::Zero(centers.size());
  for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
    centers[iCenter].setZero();
  for (int iData = 0; iData < data.rows(); iData++)
  {
    centers[assign[iData]] += data.row(iData).transpose();
    nbPoints[assign[iData]]++;
  }
  for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
    centers[iCenter] /= nbPoints[iCenter];

  // Init covars
  for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
    covars[iCenter].setZero();
  for (int iData = 0; iData < data.rows(); iData++)
    covars[assign[iData]] += (data.row(iData).transpose() - centers[assign[iData]]) * (data.row(iData).transpose() - centers[assign[iData]]).transpose();
  for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
    covars[iCenter] /= nbPoints[iCenter];

  // Be sure that covar is invertible
  for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
      covars[iCenter] += MatrixXd::Identity(covars[iCenter].rows(), covars[iCenter].cols()) * 1e-5;

  // Init priors
  for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
    priors[iCenter] = 1. / centers.size();
}

void FunctionApproximatorGMR::kMeansInit(const MatrixXd& data, std::vector<VectorXd>& centers, std::vector<double>& priors,
  std::vector<MatrixXd>& covars, int nbMaxIter)
{

  MatrixXd dataCentered = data.rowwise() - data.colwise().mean();
  MatrixXd dataCov = dataCentered.transpose() * dataCentered / data.rows();
  MatrixXd dataCovInverse = dataCov.inverse();

  std::vector<int> dataIndex;
  for (int i = 0; i < data.rows(); i++)
    dataIndex.push_back(i); 
  std::random_shuffle (dataIndex.begin(), dataIndex.end());

  for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
    centers[iCenter] = data.row(dataIndex[iCenter]);

  VectorXi assign(data.rows());
  assign.setZero();

  bool converged = false;
  for (int iIter = 0; iIter < nbMaxIter && !converged; iIter++)
  {
    // E step
    converged = true;
    for (int iData = 0; iData < data.rows(); iData++)
    {
      VectorXd v = (centers[assign[iData]] - data.row(iData).transpose());

      double minDist = v.transpose() * dataCovInverse * v;

      for (int iCenter = 0; iCenter < (int)centers.size(); iCenter++)
      {
        if (iCenter == assign[iData])
          continue;

        v = (centers[iCenter] - data.row(iData).transpose());
        double dist = v.transpose() * dataCovInverse * v;
        if (dist < minDist)
        {
          converged = false;
          minDist = dist;
          assign[iData] = iCenter;
        }
      }
    }

    // M step
    VectorXi nbPoints = VectorXi::Zero(centers.size());
    for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
      centers[iCenter].setZero();
    for (int iData = 0; iData < data.rows(); iData++)
    {
      centers[assign[iData]] += data.row(iData).transpose();
      nbPoints[assign[iData]]++;
    }
    for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
      centers[iCenter] /= nbPoints[iCenter];
  }

  // Init covars
  VectorXi nbPoints = VectorXi::Zero(centers.size());
  for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
    covars[iCenter].setZero();
  for (int iData = 0; iData < data.rows(); iData++)
  {
    covars[assign[iData]] += (data.row(iData).transpose() - centers[assign[iData]]) * (data.row(iData).transpose() - centers[assign[iData]]).transpose();
    nbPoints[assign[iData]]++;
  }
  for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
    covars[iCenter] /= nbPoints[iCenter];

  // Be sure that covar is invertible
  for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
    covars[iCenter] += MatrixXd::Identity(covars[iCenter].rows(), covars[iCenter].cols()) * 1e-5f;

  // Init priors
  for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
    priors[iCenter] = 1. / centers.size();
}

void saveGMM(string directory, const vector<VectorXd>& centers, const vector<MatrixXd>& covars, int iter)
{
  for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
  {
    stringstream stream;
    stream << "gmm_iter" << setw(2) << setfill('0') << iter << "_gauss" << iCenter << "_mu.txt";
    string filename = stream.str();
    if (!saveMatrix(directory, filename,  centers[iCenter],  true))
      exit(0);
    cout << "  filename=" << filename << endl;
    
    stringstream stream2;
    stream2 << "gmm_iter" << setw(2) << setfill('0') << iter << "_gauss" << iCenter << "_covar.txt";
    filename = stream2.str();
    cout << "  filename=" << filename << endl;
    
    if (!saveMatrix(directory, filename,  covars[iCenter],  true))
      exit(0);
    
  }
}

void FunctionApproximatorGMR::EM(const MatrixXd& data, std::vector<VectorXd>& centers, std::vector<double>& priors,
    std::vector<MatrixXd>& covars, int nbMaxIter)
{
  MatrixXd assign(centers.size(), data.rows());
  assign.setZero();

  double oldLoglik = -1e10f;
  double loglik = 0;

  for (int iIter = 0; iIter < nbMaxIter; iIter++)
  {
    // For debugging only
    //saveGMM("/tmp/demoTrainFunctionApproximators/GMR",centers,covars,iIter);
    
    // E step
    for (int iData = 0; iData < data.rows(); iData++)
      for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
        assign(iCenter, iData) = priors[iCenter] * normal(data.row(iData).transpose(), centers[iCenter], covars[iCenter]);

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
    for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
    {
      centers[iCenter].setZero();
      covars[iCenter].setZero();
      priors[iCenter] = 0;
    }

    for (int iData = 0; iData < data.rows(); iData++)
    {
      for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
      {
        centers[iCenter] += assign(iCenter, iData) * data.row(iData).transpose();
        priors[iCenter] += assign(iCenter, iData);
      }
    }

    for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
    {
      centers[iCenter] /= assign.row(iCenter).sum();
      priors[iCenter] /= assign.cols();
    }

    for (int iData = 0; iData < data.rows(); iData++)
      for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
        covars[iCenter] += assign(iCenter, iData) * (data.row(iData).transpose() - centers[iCenter]) * (data.row(iData).transpose() - centers[iCenter]).transpose();

    for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
      covars[iCenter] /= assign.row(iCenter).sum();

    // Be sure that covar is invertible
    for (size_t iCenter = 0; iCenter < centers.size(); iCenter++)
      covars[iCenter] += MatrixXd::Identity(covars[iCenter].rows(), covars[iCenter].cols()) * 1e-5f;
  }
}

double FunctionApproximatorGMR::normal(const VectorXd& data, const VectorXd& center, const MatrixXd& cov)
{
  double tmp = -1. / 2 * ((data - center).transpose() * cov.inverse() * (data - center))(0, 0);
  return 1. / sqrt(pow(2 * M_PI, center.cols()) * cov.determinant()) * exp(tmp);
}

template<class Archive>
void FunctionApproximatorGMR::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(FunctionApproximator);
}

}