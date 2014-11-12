#include <iostream>
#include <cmath>

#define EIGEN_RUNTIME_NO_MALLOC

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Cholesky>

using namespace std;
using namespace Eigen;


double normalPDF(const VectorXd& mean, const MatrixXd& covar, const VectorXd& input) {
  // -0.5 * (x-mu)^T * Sigma^-1 * (x-mu)
  double inside_exp = -0.5*(mean-input).dot(covar.inverse()*(mean-input));
  // (1/sqrt( (2*PI)^k * |Sigma| )) * exp(-0.5 * (x-mu)^T * Sigma^-1 * (x-mu)) 
  return pow(pow(2*M_PI,input.size()*covar.determinant()),-0.5)*exp(inside_exp);
}


double normalPDFRealTime(const VectorXd& mean, const MatrixXd& covar, const VectorXd& input) {
  
  // These things must be preallocated or precomputed, or they will dynamically allocate memory.
  VectorXd diff_prealloc(mean.size());
  VectorXd covar_times_diff_prealloc(mean.size());
  MatrixXd covar_inverse = covar.inverse();
  double covar_determinant = covar.determinant();
  
  Eigen::internal::set_is_malloc_allowed(false);

  // (x-mu)
  diff_prealloc = input - mean;
  // Sigma^-1 * (x-mu)
  covar_times_diff_prealloc.noalias() = covar_inverse*diff_prealloc;
  // -0.5 * (x-mu)^T * Sigma^-1 * (x-mu)
  double inside_exp = -0.5*(diff_prealloc).dot(covar_times_diff_prealloc);
  
  // 1/sqrt( (2*PI)^k * |Sigma| ) 
  double normalization = pow(pow(2*M_PI,input.size()*covar_determinant),-0.5);

  Eigen::internal::set_is_malloc_allowed(true);
  
  // (1/sqrt( (2*PI)^k * |Sigma| )) * exp(-0.5 * (x-mu)^T * Sigma^-1 * (x-mu)) 
  return normalization*exp(inside_exp);
}

int main(int n_args, char** args)
{
  VectorXd mean  = VectorXd::Zero(2);
  MatrixXd covar(2,2);
  covar << 0.9, 0.1, 0.1, 0.9;
  
  VectorXd input(2);
  input << 0.1, 0.3;
  
  cout << "  mean=" << endl << mean << endl;  
  cout << "  covar=" << endl << covar << endl;
  cout << "  input=" << endl << input << endl;
  
  cout << "  normalPDF(mean,covar,input)=" << normalPDF(mean,covar,input) << endl;
  cout << "  normalPDFRealTime(mean,covar,input)=" << normalPDFRealTime(mean,covar,input) << endl;
  
}


