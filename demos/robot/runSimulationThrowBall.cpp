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

#include "runSimulationThrowBall.hpp"

#include <fstream>
#include <iostream>

#include "dmp/Dmp.hpp"
#include "dmp/Trajectory.hpp"

using namespace std;
using namespace Eigen;
namespace DmpBbo {

ThrowBallSimulator::ThrowBallSimulator(void)
{
  time = 0.0;

  y_endeff = VectorXd::Zero(2);
  yd_endeff = VectorXd::Zero(2);
  ydd_endeff = VectorXd::Zero(2);

  y_ball = VectorXd::Zero(2);
  yd_ball = VectorXd::Zero(2);
  ydd_ball = VectorXd::Zero(2);

  ball_in_hand = true;

  y_floor = -0.3;
}

void ThrowBallSimulator::integrateStep(double dt, Eigen::VectorXd y_des,
                                       Eigen::VectorXd yd_des,
                                       Eigen::VectorXd ydd_des)
{
  // Simple version without dynamics for now: end_eff = end_eff_des
  y_endeff = y_des;
  yd_endeff = yd_des;
  ydd_endeff = ydd_des;

  if (ball_in_hand) {
    // If the ball is in your hand, it moves along with your hand
    y_ball = y_endeff;
    yd_ball = yd_endeff;
    ydd_ball = ydd_endeff;

    if (time > 0.6) {
      // Release the ball to throw it!
      ball_in_hand = false;
    }
  } else  // ball_in_hand is false => ball is flying through the air
  {
    ydd_ball(0) = 0.0;
    ydd_ball(1) = -9.81;  // Gravity

    // Euler integration
    yd_ball = yd_ball + dt * ydd_ball;
    y_ball = y_ball + dt * yd_ball;

    if (y_ball(1) < y_floor) {
      // Ball hits the floor
      y_ball(1) = y_floor;
      // No more movement
      yd_ball = VectorXd::Zero(2);
      ydd_ball = VectorXd::Zero(2);
    }
  }

  time += dt;
  // if x(t_i-1,BALL_IN_CUP)
  //  % If the ball is in the cup, it does not move
  //  x(t_i,BALL_X:BALL_Y) = x(t_i-1,BALL_X:BALL_Y);
  //  x(t_i,BALL_IN_CUP) = 1; % Once in the cup, always in the cup
  //
  // else
  //
  //  if x(t_i,HOLD_BALL)
  //    % If the ball is in your hand, it moves along with your hand
  //    x(t_i,BALL_X:BALL_Y) = x(t_i,REF_X:REF_Y);
  //    x(t_i,BALL_XD) = diff(x([t_i-1 t_i],BALL_X))/dt;
  //    x(t_i,BALL_YD) = diff(x([t_i-1 t_i],BALL_Y))/dt;
  //
  //  else
  //    % If the ball is not in your hand, it simply falls
  //    x(t_i,BALL_XDD) = 0;
  //    x(t_i,BALL_YDD) = -g;
  //
  //    % Euler integration
  //    x(t_i,BALL_XD:BALL_YD) = x(t_i-1,BALL_XD:BALL_YD) +
  //    dt*x(t_i,BALL_XDD:BALL_YDD); x(t_i,BALL_X:BALL_Y) =
  //    x(t_i-1,BALL_X:BALL_Y) + dt*x(t_i,BALL_XD:BALL_YD);
  //
  //  end
}

int ThrowBallSimulator::getStateSize(void) { return 1 + 6 * 2 + 1; }

Eigen::VectorXd ThrowBallSimulator::getState(void)
{
  VectorXd state(getStateSize());
  state(0) = time;
  state.segment(1, 2) = y_endeff;
  state.segment(3, 2) = yd_endeff;
  state.segment(5, 2) = ydd_endeff;
  state.segment(7, 2) = y_ball;
  state.segment(9, 2) = yd_ball;
  state.segment(11, 2) = ydd_ball;
  state(13) = (ball_in_hand ? 1.0 : 0.0);

  return state;
}

}  // namespace DmpBbo
