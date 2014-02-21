/**
 * \file testEigenBoostSerialization.cpp
 * \author Freek Stulp
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
 
#include "dmpbbo_io/EigenBoostSerialization.hpp"

#include <fstream>
#include <sstream>

#include <eigen3/Eigen/Core>

#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>


using namespace std;
using namespace Eigen;

int main(int n_args, char* args[])
{
  int dim = 3;
  
  VectorXd vector(dim), vector_out;
  vector << 0.0, 1.0, 0.5;
  MatrixXd matrix(dim,dim), matrix_out;
  matrix << 3.0, 0.5, 1.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.3;
  MatrixXd matrix_diag(dim,dim);
  matrix_diag << 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2;
 
  string filename("/tmp/matrix.xml"); 
  for (int i=0; i<3; i++)
  {
    MatrixXd m_in, m_out;
    switch (i)
    {
    case 0: m_in = vector; break;
    case 1: m_in = matrix; break;
    case 2: m_in = matrix_diag; break;
    }
  
    // create and open a character archive for output
    std::ofstream ofs(filename);
    boost::archive::xml_oarchive oa(ofs);
    oa << BOOST_SERIALIZATION_NVP(m_in);
    ofs.close();

    std::ifstream ifs(filename);
    boost::archive::xml_iarchive ia(ifs);
    ia >> BOOST_SERIALIZATION_NVP(m_out);
    ifs.close();
  
    cout << "___________________________________________" << endl;
    cout << m_in << endl << endl;
    cout << m_out << endl;
  }
  
  
}


