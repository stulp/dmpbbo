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

#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/ModelParametersLWR.hpp"
#include "bbo/updaters/UpdaterCovarAdaptation.hpp"
#include "bbo/DistributionGaussian.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"

#include <vector>
#include <string>
#include <eigen3/Eigen/Core>

#include <iostream>
#include <fstream>

using namespace std;

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


class PyDmpBbo {
public:
    PyDmpBbo();
    void run(double tau, int n_time_steps, int n_basis_functions, int input_dim, double intersection, const std::string &save_dir);

private:
    ;
};

#endif
