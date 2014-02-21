/**
 * \file testTrajectory.cpp
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

#include "dmp/Trajectory.hpp"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

int main(int n_args, char** args)
{
  VectorXd ts = VectorXd::LinSpaced(11,0,0.5  );
  VectorXd y_first(2); y_first << 0.0,1.0;
  VectorXd y_last(2);  y_last  << 0.4,0.5;
  Trajectory traj = Trajectory::generateMinJerkTrajectory(ts, y_first, y_last);

  string filename("/tmp/testTrajectory.txt");
  
  ofstream outfile;
  outfile.open(filename.c_str()); 
  outfile << traj; 
  outfile.close();
  
  Trajectory traj_reread = Trajectory::readFromFile(filename);

  cout << "__________________" << endl;
  cout << traj;
  cout << "__________________" << endl;
  cout << traj_reread;
  
  MatrixXd misc = RowVectorXd::LinSpaced(3,1,3);
  traj.set_misc(misc);
  
  filename = string("/tmp/testTrajectoryMisc.txt");
  
  outfile.open(filename.c_str()); 
  outfile << traj; 
  outfile.close();
  
  Trajectory traj_reread_misc = Trajectory::readFromFile(filename);

  cout << "__________________" << endl;
  cout << traj;
  cout << "__________________" << endl;
  cout << traj_reread_misc;

  // create and open a character archive for output
  std::string filename_xml("/tmp/trajectory.xml");

  std::ofstream ofs(filename_xml);
  boost::archive::xml_oarchive oa(ofs);
  oa << boost::serialization::make_nvp("traj",traj);
  ofs.close();

  std::ifstream ifs(filename_xml);
  boost::archive::xml_iarchive ia(ifs);
  Trajectory traj_out;
  ia >> BOOST_SERIALIZATION_NVP(traj_out);
  ifs.close();
  
  cout << "___________________________________________" << endl;
  cout << "  filename_xml=" << filename_xml << endl;
  cout << traj << endl;
  cout << traj_out << endl;

  return 0;
}







