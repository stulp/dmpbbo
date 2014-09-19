/**
 * @file EigenFileIO.tpp
 * @brief  Source file for input/output of Eigen matrices to ASCII files.
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


/** Load an Eigen matrix from an ASCII file.
 * \param[in] filename Name of the file from which to read the matrix
 * \param[out] m The matrix that was read from file
 * \return true if loading was successful, false otherwise
 */ 
template<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
inline bool loadMatrix(std::string filename, Eigen::Matrix<Scalar,RowsAtCompileTime,ColsAtCompileTime>& m)
{
  
  // General structure
  // 1. Read file contents into vector<double> and count number of lines
  // 2. Initialize matrix
  // 3. Put data in vector<double> into matrix
  
  std::ifstream input(filename.c_str());
  if (input.fail())
  {
    std::cerr << "ERROR. Cannot find file '" << filename << "'." << std::endl;
    m = Eigen::Matrix<Scalar,RowsAtCompileTime,ColsAtCompileTime>(0,0);
    return false;
  }
  std::string line;
  Scalar d;
  
  std::vector<Scalar> v;
  int n_rows = 0;
  while (getline(input, line))
  {
    ++n_rows;
    std::stringstream input_line(line);
    while (!input_line.eof())
    {
      input_line >> d;
      v.push_back(d);
    }
  }
  input.close();
  
  int n_cols = v.size()/n_rows;
  m = Eigen::Matrix<Scalar,RowsAtCompileTime,ColsAtCompileTime>(n_rows,n_cols);
  
  for (int i=0; i<n_rows; i++)
    for (int j=0; j<n_cols; j++)
      m(i,j) = v[i*n_cols + j];
    
  return true;
}

/** Save an Eigen matrix to an ASCII file.
 * \param[in] directory Name of the directory to which to save the matrix
 * \param[in] filename Name of the file to which to save the matrix
 * \param[in] matrix The matrix to save to file
 * \param[in] overwrite Whether to overwrite any existing files
 * \return true if saving was successful, false otherwise
 */ 
template<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
inline bool saveMatrix(std::string directory, std::string filename, Eigen::Matrix<Scalar,RowsAtCompileTime,ColsAtCompileTime> matrix, bool overwrite)
{
  if (directory.empty())
    return false;

  
  if (!boost::filesystem::exists(directory))
  {
    // Directory doesn't exist. Try to create it.
    if (!boost::filesystem::create_directories(directory))
    {
      std::cerr << "Couldn't make directory '" << directory << "'. Not saving data." << std::endl;
      return false;
    }
  }
  
  filename = directory+"/"+filename;
  saveMatrix(filename,matrix,overwrite);
  
  return true;
}

/** Save an Eigen matrix to an ASCII file.
 * \param[in] filename Name of the file to which to save the matrix
 * \param[in] matrix The matrix to save to file
 * \param[in] overwrite Whether to overwrite any existing files
 * \return true if saving was successful, false otherwise
 */ 
template<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
inline bool saveMatrix(std::string filename, Eigen::Matrix<Scalar,RowsAtCompileTime,ColsAtCompileTime> matrix, bool overwrite)
{
  if (boost::filesystem::exists(filename))
  {
    if (!overwrite)
    {
      // File exists, but overwriting is not allowed. Abort.
      std::cerr << "File '" << filename << "' already exists. Not saving data." << std::endl;
      return false;
    }
  }
  
  
  std::ofstream file;
  file.open(filename.c_str());
  if (!file.is_open())
  {
    std::cerr << "Couldn't open file '" << filename << "' for writing." << std::endl;
    return false;
  }
  
  file << std::fixed;
  file << matrix;
  file.close();
  
  return true;
  
}

