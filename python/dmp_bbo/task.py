class Task:
    """Interface for cost functions, which define a task.
    For further information see the section on \ref sec_bbo_task_and_task_solver
    """
    
    def costLabels(self):
        """Labels for the different cost components.
        
        The cost function evaluateRollout may return an array of costs. The first one cost[0] is always the sum of the other ones, i.e. costs[0] = sum(costs[1:]). This function returns labels for the individual cost components.
        """
        return []

    def evaluateRollout(self,cost_vars,sample):
        """The cost function which defines the task.
       
        \param[in] cost_vars All the variables relevant to computing the cost. These are determined by TaskSolver::performRollout(). For further information see the section on \ref sec_bbo_task_and_task_solver
        \param[in] sample The sample from which the rollout was generated. Passing this to the cost function is useful when performing regularization on the sample. For further information see the section on \ref sec_bbo_task_and_task_solver
         \return costs The scalar cost components for the sample. The first item costs[0] should contain the total cost.
        """
        raise NotImplementedError('subclasses must override evaluateRollout()!')
        
    #def setRegularization(self,regularization):
    #    self.regularization = regularization
    #    
    #def regularizationCost(self,sample):
    #    # case 1: regularization is float
    #    # case 2: regularization is vector
    #    # case 3: regularization is matrix
    #    self.regularization = regularization
        
    def plotRollout(self,cost_vars,ax):
        #print("plotRollout not implemented.")
        pass
