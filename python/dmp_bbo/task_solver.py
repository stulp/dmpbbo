## \file task_solver.py
#  \author Freek Stulp
#  \brief  TaskSolver class
#  
#  \ingroup Demos
#  \ingroup DMP_BBO

## Interface for classes that can perform rollouts.
# For further information see the section on \ref sec_bbo_task_and_task_solver
class TaskSolver:
    
    ## Perform rollouts, that is, given a set of samples, determine all the variables that are relevant to evaluating the cost function. 
    # \param[in] sample The sample
    # \return The variables relevant to computing the cost.
    def performRollout(self,sample):
        raise NotImplementedError('subclasses must override performRollout()!')
        
    def plotRollout(self,cost_vars,ax):
        print("plotRollout not implemented.")

