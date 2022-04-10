/**
 * \file robotExecuteDmp.cpp
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

#include "dmp/Trajectory.hpp"

#include <iostream>
#include <fstream>

using namespace std;
using namespace Eigen;
namespace DmpBbo {

void runSimulationThrowBall(Trajectory* trajectory, MatrixXd& cost_vars)
{

  VectorXd ts = trajectory->ts();
  MatrixXd y_endeff = trajectory->ys();
  MatrixXd yd_endeff = trajectory->yds();
  MatrixXd ydd_endeff = trajectory->ydds();
  int n_time_steps = y_endeff.rows();
  MatrixXd y_ball(n_time_steps,2); 
  MatrixXd yd_ball(n_time_steps,2);
  MatrixXd ydd_ball(n_time_steps,2);
  
  
  double dt = trajectory->duration()/n_time_steps;
  
  bool ball_in_hand = true;
  
  for (int ii=0; ii<n_time_steps; ii++)
  {  
    if (ball_in_hand)
    {
      // If the ball is in your hand, it moves along with your hand
      y_ball.row(ii) = y_endeff.row(ii); 
      yd_ball.row(ii) = yd_endeff.row(ii); 
      ydd_ball.row(ii) = ydd_endeff.row(ii); 
      
      if (ts(ii)>0.6)
      {
        // Release the ball to throw it!
        ball_in_hand = false;
      }
    }
    else // ball_in_hand is false => ball is flying through the air
    {
        ydd_ball(ii,0) = 0.0;
        ydd_ball(ii,1) = -9.81; // Gravity
        
        // Euler integration
        yd_ball.row(ii) = yd_ball.row(ii-1) + dt*ydd_ball.row(ii);
        y_ball.row(ii) = y_ball.row(ii-1) + dt*yd_ball.row(ii);
        
        if (y_ball(ii,1)<-0.3)
        {
          // Ball hits the floor (floor is at -0.3)
          y_ball(ii,1) = -0.3;
          yd_ball.row(ii) = VectorXd::Zero(2);
          ydd_ball.row(ii) = VectorXd::Zero(2);
          
        }
    }
    
    //if x(t_i-1,BALL_IN_CUP)
    //  % If the ball is in the cup, it does not move
    //  x(t_i,BALL_X:BALL_Y) = x(t_i-1,BALL_X:BALL_Y);
    //  x(t_i,BALL_IN_CUP) = 1; % Once in the cup, always in the cup
    //  
    //else
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
    //    x(t_i,BALL_XD:BALL_YD) = x(t_i-1,BALL_XD:BALL_YD) + dt*x(t_i,BALL_XDD:BALL_YDD);
    //    x(t_i,BALL_X:BALL_Y) = x(t_i-1,BALL_X:BALL_Y) + dt*x(t_i,BALL_XD:BALL_YD);
    //    
    //  end
  }
  
  trajectory->set_misc(y_ball);
  
  trajectory->asMatrix(cost_vars);
}

}
