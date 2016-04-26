class CostFunction:
    """ Interface for cost functions, which define a cost_function.
    For further information see the section on \ref sec_bbo_task_and_task_solver
    """

    def evaluate(self,sample):
        """The cost function which defines the cost_function.
        
         \param[in] sample The sample
         \return costs The scalar cost components for the sample. The first item costs[0] should contain the total cost.
        """
        raise NotImplementedError('subclasses must override evaluate()!')

