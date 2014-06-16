#include "pydmpbbo.h"

using namespace std;
using namespace Eigen;

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

DmpBbo::Trajectory getDemoTrajectory(const VectorXd& ts);

PyDmpBbo::PyDmpBbo() {
    ;
}

void PyDmpBbo::run(double tau, int n_time_steps, int n_basis_functions, int input_dim, double intersection, const std::string &save_dir) {

  std::string save_directory(save_dir);

  // GENERATE A TRAJECTORY 

  VectorXd ts = VectorXd::LinSpaced(n_time_steps,0,tau); // Time steps
  DmpBbo::Trajectory trajectory = getDemoTrajectory(ts); // getDemoTrajectory() is implemented below main()
  int n_dims = trajectory.dim();

  
  // MAKE THE FUNCTION APPROXIMATORS
  
  // Initialize some meta parameters for training LWR function approximator

  DmpBbo::MetaParametersLWR* meta_parameters = new DmpBbo::MetaParametersLWR(input_dim,n_basis_functions,intersection);      
  DmpBbo::FunctionApproximatorLWR* fa_lwr = new DmpBbo::FunctionApproximatorLWR(meta_parameters);  
  
  // Clone the function approximator for each dimension of the DMP
  vector<DmpBbo::FunctionApproximator*> function_approximators(n_dims);    
  for (int dd=0; dd<n_dims; dd++)
    function_approximators[dd] = fa_lwr->clone();
  
  // CONSTRUCT AND TRAIN THE DMP
  
  // Initialize the DMP
  DmpBbo::Dmp* dmp = new DmpBbo::Dmp(n_dims, function_approximators, DmpBbo::Dmp::KULVICIUS_2012_JOINING);

  // And train it. Passing the save_directory will make sure the results are saved to file.
  dmp->train(trajectory,save_directory);

  
  // INTEGRATE DMP TO GET REPRODUCED TRAJECTORY
  
  DmpBbo::Trajectory traj_reproduced;
  double tau_repro = 0.9;
  int n_time_steps_repro = 91;
  ts = VectorXd::LinSpaced(n_time_steps_repro,0,tau_repro); // Time steps
  dmp->analyticalSolution(ts,traj_reproduced);

  // Integrate again, but this time get more information
  MatrixXd xs_ana, xds_ana, forcing_terms_ana, fa_output_ana;
  dmp->analyticalSolution(ts,xs_ana,xds_ana,forcing_terms_ana,fa_output_ana);

  
  // WRITE THINGS TO FILE
  
  bool overwrite = true;
  
  trajectory.saveToFile(save_directory,"demonstration_traj.txt",overwrite);
  traj_reproduced.saveToFile(save_directory,"reproduced_traj.txt",overwrite);
    
  MatrixXd output_ana(ts.size(),1+xs_ana.cols()+xds_ana.cols());
  output_ana << xs_ana, xds_ana, ts;
  DmpBbo::saveMatrix(save_directory,"reproduced_xs_xds.txt",output_ana,overwrite);
  DmpBbo::saveMatrix(save_directory,"reproduced_forcing_terms.txt",forcing_terms_ana,overwrite);
  DmpBbo::saveMatrix(save_directory,"reproduced_fa_output.txt",fa_output_ana,overwrite);

  delete meta_parameters;
  delete fa_lwr;
  delete dmp;
}


DmpBbo::Trajectory getDemoTrajectory(const VectorXd& ts)
{
  bool use_viapoint_traj= true;
  if (use_viapoint_traj)
  {
    int n_dims = 1;
    VectorXd y_first = VectorXd::Zero(n_dims);
    VectorXd y_last  = VectorXd::Ones(n_dims);
    double viapoint_time = 0.25;
    double viapoint_location = 0.5;
  
    VectorXd y_yd_ydd_viapoint = VectorXd::Zero(3*n_dims);
    y_yd_ydd_viapoint.segment(0*n_dims,n_dims).fill(viapoint_location); // y         
    return  DmpBbo::Trajectory::generatePolynomialTrajectoryThroughViapoint(ts,y_first,y_yd_ydd_viapoint,viapoint_time,y_last); 
  }
  else
  {
    int n_dims = 2;
    VectorXd y_first = VectorXd::LinSpaced(n_dims,0.0,0.7); // Initial state
    VectorXd y_last  = VectorXd::LinSpaced(n_dims,0.4,0.5); // Final state
    return DmpBbo::Trajectory::generateMinJerkTrajectory(ts, y_first, y_last);
  }

}
