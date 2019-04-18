Step-by-step howto for training and optimizing a DMP on a real robot
===============

*It is assumed that you have already read the tutorials on <a href="dmp_bbo.md">Black Box Optimization of Dynamical Movement Primitives</a>.* 

This tutorial will describe the steps involved in training and optimizing a DMP on a real robot. Probably the easiest way to get dmpbbo running for your robot is to copy this directory `cp -uva dmp_bbo_robot my_own_optimization`, and adapt the cpp and py files to your robot and task.

## Step 1: Train the DMP with a demonstration

It is common practice to initialize the DMP with a demonstrated trajectory, so that the optimization does not have to start from scratch. Given that the optimization algorithms are local, such an initialization is essential to avoid local minima that do not solve the task.

This initialization is done with the following command (the binary is compiled from `step1A_trainDmpFromTrajectoryFile.cpp`):

    ./step1A_trainDmpFromTrajectoryFile trajectory.txt results/dmp.xml results/policy_parameters.txt results/train/ 6

This reads a trajectory from a file (for the format see the Trajectory class), trains a DMP, and saves it to an xml file with boost::serialization. The policy parameters of this DMP, which are to be optimized in the subsequent optimization step, are also stored in `policy_parameters.txt`. To analyze the fitting process and tune training meta-parameters, the intermediate fitting results are written to `results/train/`. These can be plotted with:

     python3 step1B_trainDmpFromTrajectoryFilePlot.py results/train/

The "6" in the call to `step1A_trainDmpFromTrajectoryFile` is a meta-parameter, in this example the number of basis functions. You can try different values and plot the results until a good fit is achieved.

## Step 2: Define the task (i.e. cost function) and implement executing rollouts on the robot

Defining the task requires you to make a class that inherits from `dmp_bbo.Task`, and implements the following functions:

* `evaluateRollout(cost_vars,...)`. This is the cost function, which takes the cost-relevant variables (cost_vars) as an input, and returns the cost associated with the rollout. cost_vars thus defines the variables the robot needs to record when performing a rollout, as these variables are required to compute the cost.
* `plotRollout(cost_vars,...)`. This function visualizes one rollout.

An example is Python script is available, which writes the defined task to a directory.

    python3 step1_defineTask.py results/

The task converts cost-relevant variables into a cost. The robot, who is responsible for executing the rollouts, should write the cost-relevant variable to a file. Therefore, the used must write an interface to the robot that reads a dmp, executes it, and writes the results to a file containing the cost-relevant variables. In the dmp_bbo_robot examples, this interface is the executable `robotPerformRollout` (compiled from `robotPerformRollout.cpp`). 

    ./robotPerformRollout results/dmp.xml results/cost_vars_demonstrated.txt

The results of performing a rollout can be visualized as follows:
    
    python3 plotRollout.py results/cost_vars_demonstrated.txt results/task.p
    
This uses the `plotRollouts(cost_vars,...)` function in the `Task` to plot the rollout.
    
## Step 3: Tune the exploration noise for the optimization

During the stochastic optimization, the parameters of the DMP will be sampled from a Gaussian distribution (which parameters these are is set through the `Parameterizable` class from which `Dmp` inherits. See the "`set<string> parameters_to_optimize`" code in `step1A_trainDmpFromTrajectoryFile.cpp` for an example). The mean of this distribution will be the parameters that resulted from training the DMP with a demonstration through supervised learning. 

The covariance matrix of this distributions determines the magnitude of exploration. It should not be too low, otherwise the stochasticity of the exploration may be smaller than that of the robot movement itself, and no learning can take place. It should also not be too high for safety reasons; your robot may reach acceleration limits, joint limits, or unexpectedly bump into the environment. 

You can tune this parameter by calling the following three scripts for different exploration magnitudes:

    MAG=0.1      # Exploration magnitude to try (start low!)
    N_SAMPLES=12 # Number of samples to generate
    # Generate samples with this magnitude
    # This will save samples to directories 
    #     results/tune_exploration_0.1/rollout001/policy_parameters.txt
    #     results/tune_exploration_0.1/rollout002/policy_parameters.txt
    #     etc
    python3 step3A_tuneExploration.py results/policy_parameters.txt results/distribution_initial_covar.txt results/tune_exploration_${MAG}/ ${MAG} ${N_SAMPLES}
    # Execute the Dmps with sampled parameters on the robot
    ./step3B_performExplorationRollouts.bash results/dmp.xml results/tune_exploration_${MAG}/
    # Plot the rollouts to see the variance in the movements
    python3 step3C_tuneExplorationPlot.py results/tune_exploration_${MAG}/ results/task.p


## Step 4: Run the optimization (step by step)

Now we have trained a dmp (stored in `dmp.xml`), specified the task (stored in `task.p`), and tuned the exploration (stored in `distribution_initial_covar.txt`). Now it's time to run the optimization! This is an iterative process with two main steps (and an optional step of plotting intermediate results). Each iteration is called an "update", as it involves one update of the policy parameters.

### Step 4A: Update parameters 

This is a highly automized process, which is called as follows

    python3 step4A_oneOptimizationUpdate.py  results/

This will automatically find the most recent update (e.g. `results/update0083/`) and read all cost_vars in the rollouts in this update directory (which are stored in `update0083/rollout001/cost_vars.txt`, `update0083/rollout002/cost_vars.txt`, etc.). It then computes the costs from each cost_vars (with `task.evaluateRollout(...)`), and updates the policy parameters. Finally, it samples new policy parameters, and saves them in a new update directory (i.e. `update0084/rollout001/policy_parameters.txt`, `update0084/rollout002/policy_parameters.txt`, etc.)

Note: on the first call this script only writes the samples, but does not read the rollouts, as there are none yet.

### Step 4B: Perform rollouts

Performing the rollouts on the robot is done with the same `./robotPerformRollout` executable as above. There is a convenience bash script

    robotPerformRollouts.bash results/dmp.xml results/update00084/

which loops over all `rolloutNNNN/` directories and calls `./robotPerformRollout` on each. Finally, the `step4B_performRollouts.bash` determines the current update (e.g. `update0084/`), calls `robotPerformRollouts.bash` with this directory

    ./step4B_performRollouts.bash results/

Note that all of the scripts/programs in Step 4B will be very specific to your robot. For instance, you may have a Simulink model that implements the policy, and instead of robotPerformRollouts.bash you may have a python script or some ROS-based solution. As long as it sticks to the conventions in the directory structure with updates in `update00084/` directories, rollouts in `rollout001/`, policy parameters read from `policy_parameters.txt` and cost-relevant variables written to `cost_vars.txt` in these directories, all is good.

### Step 4C: Plotting intermediate results

Iteratively executing the two steps above iteratively leads to (you'll probably have this scripted somehow)

    python3 step4A_oneOptimizationUpdate.py results/ 
    ./step4B_performRollouts.bash results/
    python3 step4A_oneOptimizationUpdate.py results/
    ./step4B_performRollouts.bash results/
    python3 step4A_oneOptimizationUpdate.py results/
    ./step4B_performRollouts.bash results/

If you are curious about intermediate results, you can visualize them with

    python3 step4C_plotOptimization.py results/

This will automatically determine what the last update directory is.

## Design rationale

Sockets vs txt files. Keep it simple!

Approach not optimized for running millions of rollouts on a CPU cluster, but for running 10s/100s of rollouts on a real robot. 

Overhead of using txt files in terms of execution time neglible. But very nice to have a format that is human readable. Easy to adapt to different robots (whatever robot, operating system and programming language you use, they should be able to read ASCII files)

