/**
 * \file testUpdaters.cpp
 * \author Freek Stulp
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

#include "bbo/updaters/UpdaterMean.hpp"
#include "bbo/updaters/UpdaterCovarDecay.hpp"
#include "bbo/updaters/UpdaterCovarAdaptation.hpp"

#include <eigen3/Eigen/Core>

#include <iomanip> 
#include <fstream> 

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

int main()
{
  
  int n_dims = 2;
  VectorXd mean(n_dims);
  MatrixXd covar = MatrixXd::Zero(n_dims,n_dims);
  for (int ii=0; ii<n_dims; ii++)
  {
    mean(ii) = 1+ii;
    covar(ii,ii) = 0.5*(ii+1);
  }

  // MAKE THE DIFFERENT UPDATERS
  
  Updater* updaters[3];
  
  double eliteness = 10;
  string weighting_method = "PI-BB";
  updaters[0] = new UpdaterMean(eliteness,weighting_method);

  double covar_decay_factor = 0.9;
  updaters[1] = new UpdaterCovarDecay(eliteness,covar_decay_factor,weighting_method);
  
  VectorXd base_level = VectorXd::Constant(n_dims,0.01);
  bool diag_only = true;
  double learning_rate = 1.0;
  updaters[2] = new UpdaterCovarAdaptation(eliteness,weighting_method,base_level,diag_only,learning_rate);  

  DistributionGaussian distribution(mean, covar); 
  int n_samples = 10;
  MatrixXd samples(n_samples,n_dims);
  distribution.generateSamples(n_samples, samples);
  
  // Distance to origin
  VectorXd costs = samples.array().pow(2).rowwise().sum().sqrt();
    
  for (int i_updater=0; i_updater<3; i_updater++)
  {
    cout << "___________________________________________________" << endl;
    
    // Reset to original distribution for each updater
    distribution.set_mean(mean);
    distribution.set_covar(covar);
  
    VectorXd weights;
    UpdaterMean* updater_mean = dynamic_cast<UpdaterMean*>(updaters[i_updater]);
    updater_mean->costsToWeights(costs, weighting_method, eliteness, weights);
    
    cout << setprecision(3);
    cout << setw(10);
    cout << "  samples      = " << samples.transpose() << endl;
    cout << "  costs        = " << costs.transpose()   << endl;
    cout << "  weights      = " << weights.transpose() << endl;
    cout << "  distribution = " << distribution << endl;
    
    updaters[i_updater]->updateDistribution(distribution, samples, costs, distribution);
    cout << "  distribution = " << distribution << endl;
    
    
    // create and open a character archive for output
    std::string filename("/tmp/updater_");
    filename += to_string(i_updater)+".xml";
    std::ofstream ofs(filename);
    boost::archive::xml_oarchive oa(ofs);
    oa << boost::serialization::make_nvp("test_updater",updaters[i_updater]);
    ofs.close();
  
    std::ifstream ifs(filename);
    boost::archive::xml_iarchive ia(ifs);
    Updater* updater_out;
    ia >> BOOST_SERIALIZATION_NVP(updater_out);
    ifs.close();
    
    cout << "___________________________________________" << endl;
    cout << filename << endl;
    //cout << *(updaters[i_updater]) << endl;
    //cout << *updater_out << endl;
    
  }
  
}


