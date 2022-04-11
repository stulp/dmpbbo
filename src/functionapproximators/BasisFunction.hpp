/**
 * @file   BasisFunction.hpp
 * @brief  BasisFunction header file.
 * @author Freek Stulp
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

/** \ingroup FunctionApproximators
 */

#ifndef _BASISFUNCTION_H_
#define _BASISFUNCTION_H_

#include <eigen3/Eigen/Core>
#include <vector>

#include "eigenutils/eigen_realtime_check.hpp"  // Include this before Eigen header files

namespace DmpBbo {

namespace BasisFunction {

namespace Gaussian {

/** Get the kernel activations for given centers, widths and inputs. This one is
 * mainly used for Gaussian Mixture Models, hence the slightly different names
 * for the function arguments. \param[in] mus The center of the basis function
 * (size: n_basis_functions X n_dims) \param[in] covars The covariance matrices
 * of the basis functions (size: n_basis_functions X n_dims X n_dims) \param[in]
 * priors The priors for each basis functions (size: n_basis_functions)
 * \param[in] inputs The input data (size: n_samples X n_dims)
 * \param[out] kernel_activations The kernel activations, computed for each of
 * the samples in the input data (size: n_samples X n_basis_functions)
 * \param[in] normalized_basis_functions Whether to normalize the basis
 * functions
 */
void activations(const std::vector<Eigen::VectorXd>& mus,
                 const std::vector<Eigen::MatrixXd>& covars,
                 std::vector<double> priors,
                 const Eigen::Ref<const Eigen::MatrixXd>& inputs,
                 Eigen::MatrixXd& kernel_activations,
                 bool normalized_basis_functions = false);

/** Get the kernel activations for given centers, widths and inputs
 * \param[in] mus The center of the basis function (size: n_basis_functions X
 * n_dims) \param[in] sigmas The width of the basis function (size:
 * n_basis_functions X n_dims) \param[in] inputs The input data (size: n_samples
 * X n_dims) \param[out] kernel_activations The kernel activations, computed for
 * each of the samples in the input data (size: n_samples X n_basis_functions)
 * \param[in] normalized_basis_functions Whether to normalize the basis
 * functions \param[in] asymmetric_kernels Whether to use asymmetric kernels or
 * not, cf MetaParametersLWR::asymmetric_kernels()
 */
void activations(const Eigen::MatrixXd& mus, const Eigen::MatrixXd& sigmas,
                 const Eigen::Ref<const Eigen::MatrixXd>& inputs,
                 Eigen::MatrixXd& kernel_activations,
                 bool normalized_basis_functions, bool asymmetric_kernels);

}  // namespace Gaussian

namespace Cosine {

/** Get the kernel activations for given angular frequencies and phases
 * \param[in] angular_frequencies Angular frequency for each dimension and each
 * cosine basis function  (n_bfs X n_input_dims) \param[in] phases  Phase of
 * each cosine basis function (n_bfs X 1) \param[in] inputs The input data
 * (size: n_samples X n_dims) \param[out] activations The activations of the
 * cosine functions, computed for each of the samples in the input data (size:
 * n_samples X n_basis_functions)
 */
void activations(const std::vector<Eigen::MatrixXd>& angular_frequencies,
                 const std::vector<Eigen::VectorXd>& phases,
                 const Eigen::Ref<const Eigen::MatrixXd>& inputs,
                 Eigen::MatrixXd& activations);

/** Get the kernel activations for given angular frequencies and phases
 * \param[in] angular_frequencies Angular frequency for each dimension and each
 * cosine basis function  (n_bfs X n_input_dims) \param[in] phases  Phase of
 * each cosine basis function (n_bfs X 1) \param[in] inputs The input data
 * (size: n_samples X n_dims) \param[out] activations The activations of the
 * cosine functions, computed for each of the samples in the input data (size:
 * n_samples X n_basis_functions)
 */
void activations(const Eigen::MatrixXd& angular_frequencies,
                 const Eigen::VectorXd& phases,
                 const Eigen::Ref<const Eigen::MatrixXd>& inputs,
                 Eigen::MatrixXd& activations);

}  // namespace Cosine

}  // namespace BasisFunction

}  // namespace DmpBbo

#endif
