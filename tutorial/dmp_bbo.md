Black Box Optimization of Dynamical Movement Primitives
===============

*It is assumed that you have already read the tutorials on <a href="bbo.md">Black Box Optimization</a> and <a href="dmp.md">Dynamical Movement Primitives</a>.* 

When applying BBO to policy improvement (e.g. optimizing a DMP on a robot), the concept of a "rollout" becomes important. A rollout is the result of executing a policy (e.g. a DMP) with a certain set of policy parameters (e.g. the parameter of the DMP). Although the search space for optimization is in the space of the policy parameters, the costs are rather determined from the rollout.

From an implementation point of view, applying BBO to policy improvement (which may execute DMPs) requires several extensions:

* Running multiple optimizations in parallel, one for each DOF of the DMP
* Using rollouts with a Task/TaskSolver instead of a CostFunction


<a name="sec_parallel_optimization"></a>
## Optimizing DMP DOFs in parallel

Consider a 7-DOF robotic arm. With a DMP, each DOF would be represented by a different dimension of a 7-D DMP (let as assume that each dimension of the DMP uses 10 basis functions). There are then two distinct approaches towards optimizing the parameters of the DMP with respect to a cost function:

* Consider the search space for optimization to be 70-D, i.e. consider all open parameters of the DMP in one search space. For this you can use <a href="runOptimizationTask_8cpp.html#ac0e202635bf044eaa92706e19d340d41" title="Run an evolutionary optimization process, see Black Box Optimization. ">runOptimizationTask()</a>, see also the demos in demoOptimizationTask.cpp and demoOptimizationDmp.cpp. The disadvantage is that reward-weighted averaging will require more samples to perform a robust update, especially for the covariance matrix, which is 70X70!
* Run a seperate optimization for each of the 7 DOFs. Thus, here we have seven 10-D search spaces (with different distributions and samples for each DOF), but using the same costs for all optimizations. This is what is proposed for instance in PI^2. This approach is not able to exploit covariance between different DOFs, but in practice it works robustly. For this you must runOptimizationParallel(), see also the demo in demoOptimizationDmpParallel.cpp. The main difference is that instead of using one distribution (mean and covariance), D distributions must be used (these are stored in a std::vector), one for each DOF.


<a name="sec_bbo_task_and_task_solver"></a>
## CostFunction vs Task/TaskSolver

When the cost function has a simple structure, e.g. cost = ![alt text](formulae/form_38.png "$ x^2 $")  it is convenient to implement the function ![alt text](formulae/form_38.png "$ x^2 $")  in CostFunction::evaluate().

In robotics however, it is suitable to make the distinction between a task (e.g. lift an object), and an entity that solves this task (e.g. your robot, my robot, a simulated robot, etc.). For these cases, the CostFunction is split into a Task and a TaskSolver, as follows:

	CostFunction::evaluate(samples,costs) {
		TaskSolver::performRollout(samples,cost_vars)
		Task::evaluateRollout(cost_vars,costs)
	}

The roles of the Task/TaskSolver are:

* TaskSolver: performing the rollouts on the robot (based on the samples in policy space), and saving all the variables related to computing the costs into a file (called cost_vars.txt)
* Task: determining the costs from the cost-relevant variables

The idea here is that the TaskSolver uses the samples to perform a rollout (e.g. the samples represent the parameters of a policy which is executed) and computes all the variables that are relevant to determining the cost (e.g. it records the forces at the robot's end-effector, if this is something that needs to be minimized)

Some further advantages of this approach:

* Different robots can solve the exact same Task implementation of the same task.
* Robots do not need to know about the cost function to perform rollouts (and they shouldn't)
* The intermediate cost-relevant variables can be stored to file for visualization etc.
* The procedures for performing the roll-outs (on-line on a robot) and doing the evaluation/updating/sampling (off-line on a computer) can be seperated, because there is a separate TaskSolver::performRollouts function.



<a name="impl"></a>
## Implementation

When using the Task/TaskSolver approach, the optimization process is as follows:

    int n_dim = 2; // Optimize 2D problem
    
    // This is the cost function to be optimized
    CostFunction* cost_function = new CostFunctionQuadratic(VectorXd::Zero(n_dim));
    
    // This is the initial distribution
    DistributionGaussian* distribution = new DistributionGaussian(VectorXd::Random(n_dim),MatrixXd::Identity(n_dim)) 
    
    // This is the updater which will update the distribution
    double eliteness = 10.0;
    Updater* updater = new UpdaterMean(eliteness);
    
    // Some variables
    MatrixXd samples;
    VectorXd costs;
    
    for (int i_update=1; i_update<=n_updates; i_update++)
    {
      
        // 1. Sample from distribution
        int n_samples_per_update = 10;
        distribution->generateSamples(n_samples_per_update, samples);
      
        for (int i_sample=0; i_sample<n_samples_per_update; i_sample++)
        {
          // 2A. Perform the rollout
          task_solver->performRollout(sample.row(i_sample),cost_vars);
      
          // 2B. Evaluate the rollout
          task->evaluateRollout(cost_vars,costs);
          costs[i_sample] = cur_costs[0];
          
        }
      
        // 3. Update parameters
        updater->updateDistribution(*distribution, samples, costs, *distribution);
        
    }

<a name="sec_cost_components"></a>
### Cost components

evaluateRollout returns a vector of costs. This is convenient for tracing different cost components. For instance, consider a task where the costs consist of going through a viapoint with minimal acceleration. In this case, the cost components are: 1) distance to the viapoint and 2) sum of acceleration. Then 

* cost[1] = distance to the viapoint 
* cost[2] = acceleration
* cost[0] = cost[1] + cost[2] (cost[0] must always be the sum of the individual cost components)

<a name="sec_cost_vars"></a>
### Cost-relevant variables

cost_vars should contain all variables that are relevant to computing the cost. This depends entirely on the task at hand. Whatever cost_vars contains, it should be made sure that a Task and its TaskSolver are compatible, and have the same understanding of what is contained in cost_vars. Tip: for rollouts on the robot I usually let each row in cost_vars be the relevant variables at one time step. An example is given in TaskViapoint, which implements a Task in which the first N columns in cost_vars should represent a N-D trajectory. This convention is respected by TaskSolverDmp, which is able to generate such trajectories.

