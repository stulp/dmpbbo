/**
 * \file runSimulationThrowBall.cpp
 * \author Freek Stulp
 *
 * \ingroup Demos
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

#include <eigen3/Eigen/Core>
#include <fstream>

namespace DmpBbo {

// forward declaration
class Trajectory;
class Dmp;

class ThrowBallSimulator {
 public:
  ThrowBallSimulator(void);
  void integrateStep(double dt, Eigen::VectorXd y_des, Eigen::VectorXd yd_des,
                     Eigen::VectorXd ydd_des);
  int getStateSize(void);
  Eigen::VectorXd getState(void);

 private:
  double time;
  Eigen::VectorXd y_endeff;
  Eigen::VectorXd yd_endeff;
  Eigen::VectorXd ydd_endeff;
  Eigen::VectorXd y_ball;
  Eigen::VectorXd yd_ball;
  Eigen::VectorXd ydd_ball;
  bool ball_in_hand;
  double y_floor;
};

}  // namespace DmpBbo