Running things on a real robot
===============

*It is assumed that you have already read the tutorials on <a href="dmp_bbo.md">Black Box Optimization of Dynamical Movement Primitives</a>.* 


<a name="sec_bbo_one_update"></a>
## One update at a time with Task/TaskSolver

**Note.** this is currently only implemented in python/dmp_bbo/, with a demo in demos/dmp_bbo_robot

When running an optimization on a real robot, it is convenient to seperate it in two alternating steps:1

* compute the costs from the rollouts, update the parameter distribution from the costs, and generate samples from the new distribution. The input to this step are the rollouts, which should contain all the variables relevant to computing the costs. The output is the new samples. 

* perform the roll-outs, given the samples in parameter space. Save all the cost-relevant variables to file.


The first part has been implemented in Python (run_one_update.py), and the second part is specific to your robot, and can be implemented however you want.

In practice, it is not convenient to run above two phases in a loop, but rather perform them step by step. That means the optimization can be stopped at any time, which is useful if you are running long optimization (and want to go to lunch or something). An example of this update-by-update optimization is given in python/dmp_bbo/demos, where:

* demo_one_update.py performs the actual optimization

* demo_perform_rollouts.py executes the rollouts (you'll have to implement something similar on your robot)

* demo_optimization_one_by_one.bash, which alternatively calls the two scripts above.

```bash
#!/bin/bash

DIREC="/tmp/demo_optimization_one_by_one"

# Yes, I know there are for loops in bash ;-)
# But this makes it really explicit how to call the scripts
    
python demo_one_update.py $DIREC/
python demo_perform_rollouts.py $DIREC/update00001 # Replace this with your robot
    
python demo_one_update.py $DIREC/
python demo_perform_rollouts.py $DIREC/update00002 # Replace this with your robot
    
python demo_one_update.py $DIREC/
python demo_perform_rollouts.py $DIREC/update00003 # Replace this with your robot
    
python demo_one_update.py $DIREC/
python demo_perform_rollouts.py $DIREC/update00004 # Replace this with your robot
    
python demo_one_update.py $DIREC/ plotresults
```

<a name="sec_practical_howto"></a>
### Practical howto

**Todo** Update this with DMP example

Here's what you need to do to make it all work.
<a name="sec_specify_task"></a>
### Specify the task

Copy demo_one_update.py to, for instance, one_update_my_task.py. In it, replace the Task with whatever your task is. The most important function is evaluateRollout(self, cost_vars), which takes the cost-relevant variables, and returns its cost. See the sections on <a href="dmp_bbo.md#sec_cost_vars">cost-relevant variables</a> and <a href="dmp_bbo.md#sec_cost_components">cost components</a> for what these should contain.

If you want functionality for plotting a rollout, you can also implement the function plotRollout(self,cost_vars,ax), which takes costs_vars and visualizes the rollout they represent on the axis 'ax'.
<a name="sec_specify_settings"></a>
### Specify the settings of the optimization

This includes the initial distribution, the method for updating the covariance matrix and the number of samples per update. This is all set in one_update_my_task.py (your copy of demo_one_update.py).
<a name="sec_specify_robot"></a>
### Enable your robot to perform rollouts

Implement a script "performRollouts" for your robot (in whatever language you want). It should take a directory as an input. From this trajectory, it should read the file "policy_parameters.txt" and execute whatever movement it makes with these policy parameters (e.g. the parameters of the DMP, which can be set with the function setParameterVectorSelected, which it inheritst from Parameterizable)

It should write the resulting rollout to a file called cost_vars.txt (in the same directory from which policy_parameters.txt was read). This should be a vector or a matrix. What it contains depends entirely on your task. So what your robot writes to cost_vars.txt should be compatible with Task.evaluateRollout(self,cost_vars).

The "communication protocol" between run_one_update.py and the robot is very simple (conciously made so!). Only .txt files containing matrices are exchanged. An example script (in Python) can be found in demo_perform_rollouts.py. Again, this script can be anything you want (Python, bash script, Matlab, real robot), as long as it respects the format of cost_vars.txt that Task.evaluateRollout() expects. 

