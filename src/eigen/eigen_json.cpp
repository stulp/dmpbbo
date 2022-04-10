/**
 * @file   eigen_json.hpp
 * @author Freek Stulp
 *
 * This file is part of DmpBbo, a set of libraries and programs for the 
 * black-box optimization of dynamical movement primitives.
 * Copyright (C) 2022 Freek Stulp
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

#include <iostream>

#include "eigen/eigen_json.hpp"

using namespace std;
//using namespace Eigen;

namespace Eigen {
  

double from_json_to_double(const nlohmann::json& j) {
  
  if (j.contains("value")) {
    double dv = j.at("value");
    return dv;
  }
  
  // values
  if (j.contains("values")) {
    MatrixXd matrix;
    from_json(j.at("values"), matrix); 
    
    if (matrix.rows()!=1) {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "'values' should have 1 row, but has " << matrix.rows() << endl;
    }
    if (matrix.cols()!=1) {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "'values' should have 1 cols, but has " << matrix.rows() << endl;
    }
    
    return matrix(0,0);
  }

  // double
  double d = j;
  return d;
  
}


}
