import os
import sys
import numpy as np
import matplotlib.pyplot as plt

lib_path = os.path.abspath('../../python')
sys.path.append(lib_path)

from dmp_bbo.Task import Task

class TaskThrowBall(Task):
    
    def __init__(self, x_goal, x_margin, y_floor, acceleration_weight=0.0001):

        self.x_goal_ = x_goal
        self.x_margin_ = x_margin
        self.y_floor_ = y_floor
        self.acceleration_weight_ = acceleration_weight
    
    def costLabels(self):
        return ['landing site','acceleration']

    def evaluateRollout(self,cost_vars,sample):
        n_dims = 2
        n_time_steps = cost_vars.shape[0]
        
        #ts = cost_vars[:,0]
        #y = cost_vars[:,1:1+n_dims] 
        ydd = cost_vars[:,1+n_dims*2:1+n_dims*3]
        ball = cost_vars[:,-2:]
        ball_final_x = ball[-1,0]
        
        dist_to_landing_site = abs(ball_final_x-self.x_goal_)
        dist_to_landing_site -= self.x_margin_
        if dist_to_landing_site<0.0:
           dist_to_landing_site = 0.0 
        
        sum_ydd = 0.0
        if self.acceleration_weight_>0.0:
            sum_ydd = np.sum(np.square(ydd))
            
        costs = np.zeros(1+2)
        costs[1] = dist_to_landing_site
        costs[2] = self.acceleration_weight_*sum_ydd/n_time_steps
        costs[0] = np.sum(costs[1:])
        return costs
        
    def plotRollout(self,cost_vars,ax):
        """Simple script to plot y of DMP trajectory"""
        t = cost_vars[:,0]
        y = cost_vars[:,1:3]
        ball = cost_vars[:,-2:]
        
        line_handles = ax.plot(y[:,0],y[:,1],linewidth=0.5)
        line_handles_ball_traj = ax.plot(ball[:,0],ball[:,1],'-')
        line_handles_ball = ax.plot(ball[::5,0],ball[::5,1],'ok')
        plt.setp(line_handles_ball,'MarkerFaceColor','none')

        line_handles.extend(line_handles_ball_traj)

        # Plot the floor
        x_floor = [-1.0, self.x_goal_-self.x_margin_,self.x_goal_, self.x_goal_+self.x_margin_ ,0.4]
        y_floor = [self.y_floor_,self.y_floor_,self.y_floor_-0.05,self.y_floor_,self.y_floor_]
        ax.plot(x_floor,y_floor,'-k',linewidth=1)
        ax.plot(self.x_goal_,self.y_floor_-0.05,'og')
        ax.axis('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim([-0.9, 0.3])
        ax.set_ylim([-0.4, 0.3])
            
        return line_handles
