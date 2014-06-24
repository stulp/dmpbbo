#include "pydmpbbo.h"



Dmp::Dmp(int n_dims_dmp, int n_basis_functions): n_dims(n_dims_dmp), traj(), weights(MatrixXd::Constant(n_basis_functions, 1, 0.0)), function_approximators(n_dims_dmp) {
    //time_vect = VectorXd::LinSpaced(n_steps, 0.0, duration);
    VectorXd time_vect(VectorXd::LinSpaced(100, 0.0, 1));
    VectorXd min = time_vect.colwise().minCoeff();
    VectorXd max = time_vect.colwise().maxCoeff();
    meta_parameters = new DmpBbo::MetaParametersLWR(1, n_basis_functions);
    meta_parameters -> getCentersAndWidths(min, max, centers, widths);
    model_parameters = new DmpBbo::ModelParametersLWR(centers, widths, MatrixXd::Zero(n_basis_functions, 1), weights);
    //cout << "centers" << endl << centers << "widths" << widths << "weights" << weights <<endl;
    fa_lwr = new DmpBbo::FunctionApproximatorLWR(model_parameters);
    for (int dd=0; dd<n_dims_dmp; dd++) {
        function_approximators[dd] = fa_lwr->clone();
    }
    dmp = new DmpBbo::Dmp(n_dims_dmp, function_approximators, DmpBbo::Dmp::KULVICIUS_2012_JOINING);
    //dmp = new DmpBbo::Dmp(1., VectorXd::Zero(n_dims), VectorXd::Constant(n_dims, 1.), function_approximators);
    selected_labels.insert("offsets");
    dmp->setSelectedParameters(selected_labels);
}

DmpBbo::Dmp& Dmp::getDmp() {
    return *dmp;
}

void Dmp::setTau(double tau) {
    dmp->set_tau(tau);
}

void Dmp::set_initial_state(const boost::python::list& state) {
    dmp->set_initial_state(listToVectorXd(state));
}

void Dmp::set_attractor_state(const boost::python::list& state) {
    dmp->set_attractor_state(listToVectorXd(state));
}

boost::python::list Dmp::trajectory(double duration, int n_steps, const boost::python::list& weights_) {
    VectorXd ts(VectorXd::LinSpaced(n_steps, 0.0, duration));
    weights = listToVectorXd(weights_);
    //for (int dd=0; dd<n_dims; dd++) {
    dmp->setParameterVectorSelected(weights);
    //}
    dmp->analyticalSolution(ts, traj);
    boost::python::list res;
    res.append(vectorXdToList(ts));
    res.append(matrixXdToList(traj.ys()));
    res.append(matrixXdToList(traj.yds()));
    res.append(matrixXdToList(traj.ydds()));
    return res;
}

void Dmp::train(const boost::python::list& ts, const boost::python::list& xs, const boost::python::list& xds, const boost::python::list& xdds) {
    DmpBbo::Trajectory traj(listToVectorXd(ts), listToMatrixXd(xs), listToMatrixXd(xds), listToMatrixXd(xdds));
    cout << "exp_in_dim: " << function_approximators[0] -> getExpectedInputDim() << endl;
    dmp->train(traj);
}

// boost::python::list Dmp::test(const boost::python::list& _ts, const boost::python::list& weights_) {
//     VectorXd ts(listToVectorXd(_ts));
//     MatrixXd weights(listToVectorXd(weights_));
//     //for (int dd=0; dd<n_dims; dd++) {
//     dmp->setParameterVectorSelected(weights);
//     //}
//     dmp->analyticalSolution(ts, traj);
//     boost::python::list res;
//     VectorXd fa_input_phase;
//     MatrixXd f_target;
//     dmp->computeFunctionApproximatorInputsAndTargets(traj, fa_input_phase, f_target);
//     res.append(vectorXdToList(fa_input_phase));
//     res.append(matrixXdToList(f_target));
//     return res;
// }


UpdaterCovarAdaptation::UpdaterCovarAdaptation(double eliteness, string weighting_method, const boost::python::list& _base_level, bool diag_only, double learning_rate, const boost::python::list _init_mean, const boost::python::list _init_covar) {
    int dim =boost::python::len(_init_mean);
    VectorXd base_level(dim);
    VectorXd init_mean(dim);
    MatrixXd init_covar(dim, dim);
    for (int i=0; i < boost::python::len(_base_level); i++) {
        base_level(i) = boost::python::extract<double>(_base_level[i]);
    }
    for (int i=0; i < boost::python::len(_init_mean); i++) {
        init_mean[i] = boost::python::extract<double>(_init_mean[i]);
    }
    for (int i=0; i < boost::python::len(_init_covar); i++) {
        for (int j=0; j < boost::python::len(_init_covar); j++) {
            init_covar(i, j) = boost::python::extract<double>(_init_covar[i][j]);
        }
    }
    gaussian = new DmpBbo::DistributionGaussian(init_mean, init_covar);
    updater = new DmpBbo::UpdaterCovarAdaptation(eliteness,weighting_method,base_level,diag_only,learning_rate);
}

void UpdaterCovarAdaptation::updateDistribution(const boost::python::list& _samples, const boost::python::list& _costs) {
    MatrixXd samples(boost::python::len(_samples), boost::python::len(_samples[0]));
    VectorXd costs(boost::python::len(_costs));
    for (int i=0; i < boost::python::len(_samples); i++) {
        for (int j=0; j < boost::python::len(_samples[0]); j++) {
            samples(i, j) = boost::python::extract<double>(_samples[i][j]);
        }
    }
    for (int i=0; i < boost::python::len(_samples); i++) {
        costs(i) = boost::python::extract<double>(_costs[i]);
    }
    updater->updateDistribution(*gaussian, samples, costs, *gaussian, update_summary);
}

boost::python::list UpdaterCovarAdaptation::getMean(){
    VectorXd mean(update_summary.distribution_new->mean());
    boost::python::list res;
    for(int i = 0; i < mean.size(); i++) {
        res.append(mean(i));
    }
    return res;
}

boost::python::list UpdaterCovarAdaptation::getCovariance(){
    MatrixXd covar(update_summary.distribution_new->covar());
    boost::python::list res;
    for(int i = 0; i < covar.rows(); i++) {
        boost::python::list row;
        for(int j = 0; j < covar.cols(); j++) {
            row.append(covar(i, j));
        }
        res.append(row);
    }
    return res;
}
