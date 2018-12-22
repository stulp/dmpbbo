# This file is part of DmpBbo, a set of libraries and programs for the 
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
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


import numpy as np
import os

class Rollout:
    
  def __init__(self,policy_parameters, cost_vars=None, cost=None):
      self.policy_parameters = policy_parameters
      self.cost_vars = cost_vars
      self.cost = cost
  
  def __str__(self):
      string = 'Rollout[';
      np.set_printoptions(precision=3,suppress=True)
      string += 'params='+str(self.policy_parameters)
      string += ',  cost_vars='+str(self.cost_vars.shape)
      string += ',  cost= '+str(self.cost)
      string += ']'
      return string
  
  def total_cost(self):
      if self.cost is not None:
          if len(self.cost)>=1:
              return self.cost[0]
      return None      
  
  def n_cost_components(self):
      if cost is not None:
          return len(self.cost)
  
  def saveToDirectory(self,directory):
      if not os.path.exists(directory):
          os.makedirs(directory)
      d = directory
      np.savetxt(d+'/policy_parameters.txt',     self.policy_parameters)
      if self.cost_vars is not None:
          np.savetxt(d+'/cost_vars.txt',     self.cost_vars)
      if self.cost is not None:
          np.savetxt(d+'/cost.txt',     self.cost)

def saveRolloutsToDirectory(directory,rollouts):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i_rollout in range(len(rollouts)):
        cur_dir = '%s/rollout%03d' % (directory, i_rollout+1)
        rollouts[i_rollout].saveToDirectory(cur_dir)

def loadRolloutFromDirectory(directory):
    try:
        policy_parameters = np.loadtxt(directory+'/policy_parameters.txt')
    except IOError:
        return None
    try:
        cost_vars = np.loadtxt(directory+'/cost_vars.txt')
    except IOError:
        cost_vars = None
    try:
        cost = np.loadtxt(directory+'/cost.txt')
    except IOError:
        cost = None
    return Rollout(policy_parameters,cost_vars,cost)
    
def loadRolloutsFromDirectory(directory):
    rollouts = []

    i_rollout = 0
    cur_dir = '%s/rollout%03d' % (directory, i_rollout+1)
    while os.path.exists(cur_dir):
        rollouts.append(loadRolloutFromDirectory(cur_dir))
        i_rollout += 1
        cur_dir = '%s/rollout%03d' % (directory, i_rollout+1)
    return rollouts







