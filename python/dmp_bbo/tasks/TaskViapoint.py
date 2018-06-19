# This file is part of DmpBbo, a set of libraries and programs for the 
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2018 Freek Stulp
# 
# DmpBbo is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# DmpBbo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

lib_path = os.path.abspath('../../../python/')
sys.path.append(lib_path)

from dmp_bbo.Task import Task


class TaskViapoint(Task):
    
    def __init__(self, viapoint, viapoint_time=None, viapoint_radius=0.0,viapoint_weight=1.0, acceleration_weight=0.0001):
        self.viapoint_ = viapoint
        self.viapoint_time_ = viapoint_time
        self.viapoint_radius_ = viapoint_radius
        self.viapoint_weight_ = viapoint_weight
        self.acceleration_weight_ = acceleration_weight
    
    def costLabels(self):
        return ['viapoint','acceleration']

    def evaluateRollout(self,cost_vars,sample):
        n_dims = self.viapoint_.shape[0]
        n_time_steps = cost_vars.shape[0]
        
        ts = cost_vars[:,0]
        dist_to_viapoint = 0.0
        if self.viapoint_weight_>0.0:
            
            if self.viapoint_time_ is None:
                # Don't compute the distance at some time, but rather get the
                # minimum distance
                y = cost_vars[:,1:1+n_dims]
                
                viapoint_repeat =  np.repeat(np.atleast_2d(self.viapoint_),n_time_steps,axis=0)
                dists = np.linalg.norm(y-viapoint_repeat,axis=1)
                dist_to_viapoint = dists.min()
                
                
            else:
                viapoint_time_step = 0
                
                while viapoint_time_step<n_time_steps and ts[viapoint_time_step]<self.viapoint_time_:
                    viapoint_time_step+=1
                    
                y_via = cost_vars[viapoint_time_step,1:1+n_dims]
                
                dist_to_viapoint = np.linalg.norm(y_via-self.viapoint_)
                
            if self.viapoint_radius_>0.0:
                # The viapoint_radius defines a radius within which the cost is
                # always 0
                dist_to_viapoint -= self.viapoint_radius_
                if dist_to_viapoint<0.0:
                    dist_to_viapoint = 0.0
        
        sum_ydd = 0.0
        if self.acceleration_weight_>0.0:
            ydd = cost_vars[:,1+n_dims*2:1+n_dims*3]
            sum_ydd = np.sum(np.square(ydd))
        
        costs = np.zeros(1+2)
        costs[1] = self.viapoint_weight_*dist_to_viapoint
        costs[2] = self.acceleration_weight_*sum_ydd/n_time_steps
        costs[0] = costs[1] + costs[2]
        return costs
        
    def plotRollout(self,cost_vars,ax):
        """Simple script to plot y of DMP trajectory"""
        n_dims = self.viapoint_.shape[0]
        t = cost_vars[:,0]
        y = cost_vars[:,1:n_dims+1]
        if n_dims==1:
            line_handles = ax.plot(t,y,linewidth=0.5)
            ax.plot(self.viapoint_time_,self.viapoint_,'o')
            if (self.viapoint_radius_>0.0):
                r = self.viapoint_radius_
                t = self.viapoint_time_
                v = self.viapoint_[0]
                ax.plot([t,t],[v+r,v-r],'-r')
                
        elif n_dims==2:
            line_handles = ax.plot(y[:,0],y[:,1],linewidth=0.5)
            ax.plot(self.viapoint_[0],self.viapoint_[1],'ro')
            if (self.viapoint_radius_>0.0):
                circle = plt.Circle(self.viapoint_,self.viapoint_radius_, color='r', fill=False)
                ax.add_artist(circle)
            ax.axis('equal')
        else:
            line_handles = []
        return line_handles
        
    def saveToFile(self,directory,filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        my_file = open(directory+"/"+filename, 'w')
        for x in self.viapoint_:
            my_file.write(str(x)+" ")
        if self.viapoint_time_:
            my_file.write(str(self.viapoint_time_)+" ")
        else:
            my_file.write(str(-1.0)+" ")
        my_file.write(str(self.viapoint_radius_)+" ")
        my_file.write(str(self.viapoint_weight_)+" ")
        my_file.write(str(self.acceleration_weight_)+" ")
        my_file.close()

if __name__=='__main__':
    counter = 0
    for n_dims in [1,2,5]:
        for viapoint_time in [0.5, None]:
            viapoint = np.linspace(0.0,1.0,n_dims)
            task = TaskViapoint(viapoint, viapoint_time)
            task.saveToFile("/tmp/demoTaskViapoint/","viapoint"+str(counter)+".txt")
            counter += 1
        
        
