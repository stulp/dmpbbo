import numpy as np
import os

class Rollout:
    
  def __init__(self,policy_parameters, cost_vars=None, cost=None):
      self.policy_parameters = policy_parameters
      self.cost_vars = cost_vars
      self.cost = cost
  
  
  def total_cost(self):
      if self.cost:
          if len(self.cost)>=1:
              return self.cost[0]
      return None      
  
  def n_cost_components(self):
      if cost:
          return len(self.cost)
  
  def saveToDirectory(self,directory):
      if not os.path.exists(directory):
          os.makedirs(directory)
      d = directory
      np.savetxt(d+'/policy_parameters.txt',     self.policy_parameters)
      if self.cost_vars!=None:
          np.savetxt(d+'/cost_vars.txt',     self.cost_vars)
      if self.cost!=None:
          np.savetxt(d+'/cost.txt',     self.cost)

def loadRolloutFromDirectory(directory):
    policy_parameters = np.loadtxt(directory+'/policy_parameters.txt')
    try:
        cost_vars = np.loadtxt(directory+'/cost_vars.txt')
    except IOError:
        cost_vars = None
    try:
        cost = np.loadtxt(directory+'/cost.txt')
    except IOError:
        cost = None
    return Rollout(policy_parameters,cost_vars,cost)

