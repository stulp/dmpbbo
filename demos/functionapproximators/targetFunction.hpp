/**
 * @file   targetFunction.hpp
 * @brief  Header file implementation of 1D and 2D target functions.
 * Useful for generating some data to train function approximators. 
 * @author Freek Stulp
 */

#ifndef TARGETFUNCTION_HPP
#define TARGETFUNCTION_HPP

#include <eigen3/Eigen/Core>

/** Target function.
 *  \param[in] n_samples_per_dim The number of samples along each dimension. 
 *  \param[in] inputs The input vector
 *  \param[out] targets The target values for that input vector.
 */
inline void targetFunction(Eigen::VectorXi n_samples_per_dim, Eigen::MatrixXd& inputs, Eigen::MatrixXd& targets)
{
  int n_dims = n_samples_per_dim.size();
  if (n_dims==1)
  {
    // 1D Function:  y =  3*e^(-x) * sin(2*x^2);
    inputs = Eigen::VectorXd::LinSpaced(n_samples_per_dim[0], 0.0, 2.0);
    targets = 3*(-inputs.col(0)).array().exp()*(2*inputs.col(0).array().pow(2)).sin();

  }
  else
  {
    // 2D Function, similar to the example and graph here: http://www.mathworks.com/help/matlab/visualize/mapping-data-to-transparency-alpha-data.html
    int n_samples = n_samples_per_dim[0]*n_samples_per_dim[1];
    inputs = Eigen::MatrixXd::Zero(n_samples, n_dims);
    Eigen::VectorXd x1 = Eigen::VectorXd::LinSpaced(n_samples_per_dim[0], -2.0, 2.0);
    Eigen::VectorXd x2 = Eigen::VectorXd::LinSpaced(n_samples_per_dim[1], -2.0, 2.0);
    for (int ii=0; ii<x1.size(); ii++)
    {
      for (int jj=0; jj<x2.size(); jj++)
      {
        inputs(ii*x2.size()+jj,0) = x1[ii];
        inputs(ii*x2.size()+jj,1) = x2[jj];
      }
    }
    targets = 2.5*inputs.col(0).array()*exp(-inputs.col(0).array().pow(2) - inputs.col(1).array().pow(2));
    
  }
}

#endif        //  #ifndef TARGETFUNCTION_HPP

