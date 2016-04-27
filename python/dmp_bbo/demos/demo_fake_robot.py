import os
import sys
import numpy as np

lib_path = os.path.abspath('../../../python/')
sys.path.append(lib_path)

from dmp_bbo.task_solver import TaskSolver
from dmp_bbo.rollout import *


def targetFunction(a, c, inputs):
    """ Target function \f$ y = a*x^2 + c \f$
    \param[in] a a in \f$ y = a*x^2 + c \f$
    \param[in] c c in \f$ y = a*x^2 + c \f$
    \param[in] inputs x in \f$ y = a*x^2 + c \f$
    \return outputs y in \f$ y = a*x^2 + c \f$
    """
    
    # Compute a*x^2 + c
    outputs = [a*x*x + c for x in inputs]
    return np.array(outputs)


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
        cost_vars = targetFunction(a,c,self.inputs)
        return cost_vars

    def plotRollout(self,cost_vars,ax):
        line_handles = ax.plot(self.inputs,cost_vars.T,linewidth=0.5)
        #ax.plot(self.inputs,self.targets,'-o',color='k',linewidth=2)
        return line_handles


def getTaskSolver():
    inputs = np.linspace(-1.5,1.5,21);
    return DemoTaskSolverApproximateQuadraticFunction(inputs)
    
def performRolloutsFakeRobot(update_dir):
    
    print('RUNNING TASK SOLVER')
    task_solver = getTaskSolver()
    
    print('  * Loading mean from "'+update_dir+'/distribution_mean.txt"')
    distribution_mean = np.loadtxt(update_dir+"/distribution_mean.txt")

    print('  * Performing eval rollout')
    cost_vars_eval = task_solver.performRollout(distribution_mean)
    rollout_eval = Rollout(distribution_mean,cost_vars_eval) 
    
    print('  * Save rollout to "'+update_dir+'/rollout_eval"')
    rollout_eval.saveToDirectory(update_dir+"/rollout_eval")
    
    print('  * Loading samples from "'+update_dir+'/samples.txt"')
    samples = np.loadtxt(update_dir+'/samples.txt')

    print('  * Performing rollouts')
    rollouts = []
    for i_sample in range(samples.shape[0]):
        sample = samples[i_sample]
        cost_vars = task_solver.performRollout(sample)
        rollouts.append(Rollout(sample,cost_vars))
        
    print('  * Save rollouts to "'+update_dir+'/"')
    saveRolloutsToDirectory(update_dir,rollouts)
    

#def performRolloutsFakeRobotParallel(update_dir):
#    
#    print('RUNNING TASK SOLVER')
#    inputs = np.linspace(-1.5,1.5,21);
#    task_solver = DemoTaskSolverApproximateQuadraticFunction(inputs)
#
#    n_parallel = np.loadtxt(update_dir+'/n_parallel.txt')
#
#    print('  * Loading mean from "'+update_dir+'/distribution_mean_dd.txt"')
#    distribution_means = []
#    for i_parallel in range(n_parallel):
#        suf = '_%02d.txt' % i_parallel
#        distribution_means.append(np.loadtxt(update_dir+"/distribution_mean"+suf))
#
#    print('  * Performing eval rollout')
#    cost_vars_eval = task_solver_parallel.performRollouts(distribution_means)
#    
#    print('  * Save cost_vars_eval to "'+update_dir+'/cost_vars_eval.txt"')
#    np.savetxt(update_dir+"/cost_vars_eval.txt",cost_vars_eval)
#    
#    distributions = []
#    for i_parallel in range(n_parallel):
#        suf = '_%02d.txt' % i_parallel
#    print('  * Loading samples from "'+update_dir+'/samples.txt"')
#    samples = np.loadtxt(update_dir+"/samples.txt")
#
#    print('  * Performing rollouts')
#    cost_vars = task_solver.performRollouts(samples)
#    
#    print('  * Save cost_vars to "'+update_dir+'/cost_vars.txt"')
#    np.savetxt(update_dir+"/cost_vars.txt",cost_vars)

if __name__=="__main__":
    # See if input directory was passed
    if (len(sys.argv)==2):
        directory = str(sys.argv[1])
    else:
        print '\nUsage: '+sys.argv[0]+' <directory>\n';
        sys.exit()
  
    performRolloutsFakeRobot(directory)

