#ifndef PY_DMPBBO_H
#define PY_DMPBBO_H

#include <boost/python.hpp>
#include <string>

#include "dmp/Dmp.hpp"
#include "dmp/Trajectory.hpp"

//#include "dynamicalsystems/DynamicalSystem.hpp"
//#include "dynamicalsystems/ExponentialSystem.hpp"
//#include "dynamicalsystems/SigmoidSystem.hpp"
//#include "dynamicalsystems/TimeSystem.hpp"
//#include "dynamicalsystems/SpringDamperSystem.hpp"

#include "functionapproximators/FunctionApproximatorRBFN.hpp"
#include "functionapproximators/MetaParametersRBFN.hpp"
#include "functionapproximators/ModelParametersRBFN.hpp"
#include "bbo/updaters/UpdaterCovarAdaptation.hpp"
#include "bbo/DistributionGaussian.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"

#include "boostpython_eigen_conversion.h"

#include <vector>
#include <string>
#include <eigen3/Eigen/Core>

#include <iostream>
#include <fstream>

using namespace std;
using namespace Eigen;

class Dmp {
public:
    Dmp(int n_dims_dmp, int n_basis_functions);
    DmpBbo::Dmp& getDmp();
    boost::python::list trajectory(double duration, int n_steps, const boost::python::list& weights);
    void setTau(double tau);
    //boost::python::list test(const boost::python::list& _ts, const boost::python::list& weights_);



private:
    int n_dims;
    set<string> selected_labels;
    DmpBbo::Trajectory traj;
    DmpBbo::Dmp* dmp;
    DmpBbo::MetaParametersRBFN* meta_parameters;
    DmpBbo::ModelParametersRBFN* model_parameters;
    MatrixXd centers, widths, weights;
    DmpBbo::FunctionApproximatorRBFN* fa_lwr;
    vector<DmpBbo::FunctionApproximator*> function_approximators;
};

class UpdaterCovarAdaptation {
public:
    UpdaterCovarAdaptation(double eliteness, std::string weighting_method, const boost::python::list& base_level, bool diag_only, double learning_rate, const boost::python::list init_mean, const boost::python::list init_covar);

    void updateDistribution(const boost::python::list& samples, const boost::python::list& costs);

boost::python::list getMean();
boost::python::list getCovariance();

private:
    DmpBbo::DistributionGaussian* gaussian;
    DmpBbo::Updater* updater;
    vector<DmpBbo::UpdateSummary> update_summaries;
    DmpBbo::UpdateSummary update_summary;
};

#endif
