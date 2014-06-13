#include "pydmpbbo.h"

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

Trajectory getDemoTrajectory(const VectorXd& ts);

PyDmpBbo::PyDmpBbo() {
    ;
}

void PyDmpBbo::run(double tau, int n_time_steps, int n_basis_functions, int input_dim, double intersection, const std::string &save_dir) {

  std::string save_directory(save_dir);

  // GENERATE A TRAJECTORY 

  VectorXd ts = VectorXd::LinSpaced(n_time_steps,0,tau); // Time steps
  Trajectory trajectory = getDemoTrajectory(ts); // getDemoTrajectory() is implemented below main()
  int n_dims = trajectory.dim();

  
  // MAKE THE FUNCTION APPROXIMATORS
  
  // Initialize some meta parameters for training LWR function approximator

  MetaParametersLWR* meta_parameters = new MetaParametersLWR(input_dim,n_basis_functions,intersection);      
  FunctionApproximatorLWR* fa_lwr = new FunctionApproximatorLWR(meta_parameters);  
  
  // Clone the function approximator for each dimension of the DMP
  vector<FunctionApproximator*> function_approximators(n_dims);    
  for (int dd=0; dd<n_dims; dd++)
    function_approximators[dd] = fa_lwr->clone();
  
  // CONSTRUCT AND TRAIN THE DMP
  
  // Initialize the DMP
  Dmp* dmp = new Dmp(n_dims, function_approximators, Dmp::KULVICIUS_2012_JOINING);

  // And train it. Passing the save_directory will make sure the results are saved to file.
  dmp->train(trajectory,save_directory);

  
  // INTEGRATE DMP TO GET REPRODUCED TRAJECTORY
  
  Trajectory traj_reproduced;
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
  saveMatrix(save_directory,"reproduced_xs_xds.txt",output_ana,overwrite);
  saveMatrix(save_directory,"reproduced_forcing_terms.txt",forcing_terms_ana,overwrite);
  saveMatrix(save_directory,"reproduced_fa_output.txt",fa_output_ana,overwrite);

  delete meta_parameters;
  delete fa_lwr;
  delete dmp;
}


Trajectory getDemoTrajectory(const VectorXd& ts)
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
    return  Trajectory::generatePolynomialTrajectoryThroughViapoint(ts,y_first,y_yd_ydd_viapoint,viapoint_time,y_last); 
  }
  else
  {
    int n_dims = 2;
    VectorXd y_first = VectorXd::LinSpaced(n_dims,0.0,0.7); // Initial state
    VectorXd y_last  = VectorXd::LinSpaced(n_dims,0.4,0.5); // Final state
    return Trajectory::generateMinJerkTrajectory(ts, y_first, y_last);
  }

}
