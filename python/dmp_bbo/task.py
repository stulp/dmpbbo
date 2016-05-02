class Task:
    """Interface for cost functions, which define a task.
    For further information see the section on \ref sec_bbo_task_and_task_solver
    """

    def evaluateRollout(self,cost_vars):
        """The cost function which defines the task.
       
        \param[in] cost_vars All the variables relevant to computing the cost. These are determined by TaskSolver::performRollout(). For further information see the section on \ref sec_bbo_task_and_task_solver
         \return costs The scalar cost components for the sample. The first item costs[0] should contain the total cost.
        """
        raise NotImplementedError('subclasses must override evaluateRollout()!')

    def plotRollout(self,cost_vars,ax):
        #print("plotRollout not implemented.")
        pass
