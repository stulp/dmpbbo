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


import os
import sys
import numpy as np

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)


from bbo.DistributionGaussian import DistributionGaussian
from bbo.updaters import UpdaterCovarDecay

from dmp_bbo.runOptimizationTask import runOptimizationTask
from dmp_bbo.Task import Task
from dmp_bbo.TaskSolver import TaskSolver

def quadraticFunction(a, c, inputs):
    """ Target function \f$ y = a*x^2 + c \f$
    \param[in] a a in \f$ y = a*x^2 + c \f$
    \param[in] c c in \f$ y = a*x^2 + c \f$
    \param[in] inputs x in \f$ y = a*x^2 + c \f$
    \return outputs y in \f$ y = a*x^2 + c \f$
    """
    
    # Compute a*x^2 + c
    outputs = [a*x*x + c for x in inputs]
    return np.array(outputs)

class DemoTaskApproximateQuadraticFunction(Task):
    """
    The task is to choose the parameters a and c such that the function \f$ y = lib_path = os.path.abspath('../')
sys.path.append(lib_path)
a*x^2 + c \f$ best matches a set of target values y_target for a set of input values x
    """

    def __init__(self, a, c, inputs, regularization_weight=0.1):
        """ Constructor
        \param[in] a a in \f$ y = a*x^2 + c \f$
        \param[in] c c in \f$ y = a*x^2 + c \f$
        \param[in] inputs x in \f$ y = a*x^2 + c \f$
        """
        self.inputs = inputs;
        self.targets = quadraticFunction(a,c,inputs);
        self.regularization_weight = regularization_weight
  
    def costLabels(self):
        return ['MSE','regularization']
  
    def evaluateRollout(self, cost_vars, sample):
        """ Cost function
        \param[in] cost_vars y in \f$ y = a*x^2 + c \f$
        \return costs Costs of the cost_vars
        """
        diff_square = np.square(cost_vars-self.targets)
        regularization = self.regularization_weight*np.linalg.norm(sample)
        costs = [0, np.mean(diff_square), regularization]
        costs[0] = sum(costs[1:])
        return costs
        
        
    def plotRollout(self,cost_vars,ax):
        line_handles = ax.plot(self.inputs,cost_vars.T,linewidth=0.5)
        ax.plot(self.inputs,self.targets,'-o',color='k',linewidth=2)
        return line_handles

class DemoTaskSolverApproximateQuadraticFunction(TaskSolver):
    """The task solver tunes the parameters a and c such that the function \f$ y = a*x^2 + c \f$ best matches a set of target values y_target for a set of input values x"""
    
    def __init__(self,inputs):
        """\param[in] inputs x in \f$ y = a*x^2 + c \f$"""
        self.inputs = inputs
  
    def performRollout(self,sample):
        """
        \param[in] samples Samples containing variations of a and c  (in  \f$ y = a*x^2 + c \f$)
        \param[in] task_parameters Ignored
        \param[in] cost_vars Cost-relevant variables, containing the predictions
        """
        a = sample[0]
        c = sample[1]
        cost_vars = quadraticFunction(a,c,self.inputs)
        return cost_vars
    
if __name__=="__main__":
  
    directory = None
    if (len(sys.argv)>1):
        directory = sys.argv[1]
        
    inputs = np.linspace(-1.5,1.5,21);
    a = 2.0
    c = -1.0
    n_params = 2

    regularization = 0.01
    task = DemoTaskApproximateQuadraticFunction(a,c,inputs,regularization)
    task_solver = DemoTaskSolverApproximateQuadraticFunction(inputs)
  
    mean_init  =  np.full(n_params,0.5)
    covar_init =  0.25*np.eye(n_params)
    distribution = DistributionGaussian(mean_init, covar_init)
    
  
    eliteness = 10
    weighting_method = 'PI-BB'
    covar_decay_factor = 0.8
    updater = UpdaterCovarDecay(eliteness,weighting_method,covar_decay_factor)
  
    n_samples_per_update = 10
    n_updates = 40
    
    import matplotlib.pyplot as plt
    fig = plt.figure(1,figsize=(15, 5))
    
    learning_curve = runOptimizationTask(task, task_solver, distribution, updater, n_updates, n_samples_per_update, fig, directory)
  
    plt.show()

