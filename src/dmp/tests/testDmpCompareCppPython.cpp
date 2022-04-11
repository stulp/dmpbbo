/**
 * \author Freek Stulp
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

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>
#include <string>

#include "dmp/Dmp.hpp"
#include "dmp/Trajectory.hpp"
#include "eigenutils/eigen_file_io.hpp"

using namespace std;
using namespace DmpBbo;
using namespace Eigen;
using namespace nlohmann;

int main(int n_args, char** args)
{
  if (n_args != 2) cout << "Usage: " << args[0] << " <directory>" << endl;

  string directory = string(args[1]);
  string filename_dmp = directory + "/dmp.json";

  ifstream file(filename_dmp);
  json j = json::parse(file);
  // cout << j << endl;

  Dmp* dmp = j.get<Dmp*>();
  cout << *dmp << endl;

  VectorXd ts;
  if (!loadMatrix(directory + "/ts.txt", ts)) return -1;

  MatrixXd xs, xds, forcing_terms, fa_output;
  dmp->analyticalSolution(ts, xs, xds, forcing_terms, fa_output);
  Trajectory traj_reproduced;
  dmp->statesAsTrajectory(ts, xs, xds, traj_reproduced);

  MatrixXd output_ana(ts.size(), 1 + xs.cols() + xds.cols());
  output_ana << ts, xs, xds;
  bool overwrite = true;
  saveMatrix(directory, "cpp_ts_xs_xds.txt", output_ana, overwrite);
  saveMatrix(directory, "cpp_forcing_terms.txt", forcing_terms, overwrite);
  saveMatrix(directory, "cpp_fa_output.txt", fa_output, overwrite);

  traj_reproduced.saveToFile(directory, "cpp_reproduced.txt", overwrite);

  return 0;
}
