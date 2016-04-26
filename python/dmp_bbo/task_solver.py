class TaskSolver:
    """Interface for classes that can perform rollouts.
    For further information see the section on \ref sec_bbo_task_and_task_solver
    """
    
    def performRollout(self,sample):
        """ Perform rollouts, i.e. given a set of samples, determine all the variables that are relevant to evaluating the cost function. 
        \param[in] sample The sample
        \return cost_vars The variables relevant to computing the cost.
        \todo Compare to other functions
        """
        raise NotImplementedError('subclasses must override performRollout()!')
        
    def plotRollout(self,cost_vars,ax):
        print("plotRollout not implemented.")

