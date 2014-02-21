/**
 * @file EigenFileIO.hpp
 * @brief  Header file for input/output of Eigen matrices to ASCII files.
 * @author Freek Stulp
 *
 * Implementations are in the tpp file
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

#ifndef EIGENSAVEMATRIX_HPP
#define EIGENSAVEMATRIX_HPP

#include <string>
#include <fstream>
#include <iostream>

#include <boost/filesystem.hpp>

#include <eigen3/Eigen/Core>

namespace DmpBbo {

/** Load an Eigen matrix from an ASCII file.
 * \param[in] filename Name of the file from which to read the matrix
 * \param[out] m The matrix that was read from file
 * \return true if loading was successful, false otherwise
 */ 
template<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
bool loadMatrix(std::string filename, Eigen::Matrix<Scalar,RowsAtCompileTime,ColsAtCompileTime>& m);

/** Save an Eigen matrix to an ASCII file.
 * \param[in] filename Name of the file to which to save the matrix
 * \param[in] matrix The matrix to save to file
 * \param[in] overwrite Whether to overwrite any existing files
 * \return true if saving was successful, false otherwise
 */ 
template<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
bool saveMatrix(std::string filename, Eigen::Matrix<Scalar,RowsAtCompileTime,ColsAtCompileTime> matrix, bool overwrite=false);

/** Save an Eigen matrix to an ASCII file.
 * \param[in] directory Name of the directory to which to save the matrix
 * \param[in] filename Name of the file to which to save the matrix
 * \param[in] matrix The matrix to save to file
 * \param[in] overwrite Whether to overwrite any existing files
 * \return true if saving was successful, false otherwise
 */ 
template<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
bool saveMatrix(std::string directory, std::string filename, Eigen::Matrix<Scalar,RowsAtCompileTime,ColsAtCompileTime> matrix, bool overwrite=false);

#include "EigenFileIO.tpp"

}

#endif        //  #ifndef EIGENSAVEMATRIX_HPP

