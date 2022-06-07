Step-by-step howto for training and optimizing a DMP on a real robot
===============

*It is assumed that you have already read the tutorials on <a href="../tutorial/dmp_bbo.md">Black Box Optimization of Dynamical Movement Primitives</a>.* 

This tutorial will describe the steps involved in training and optimizing a DMP on a real robot. Probably the easiest way to get dmpbbo running for your robot is to copy this directory `cp -uva demos/robot mycode`, and adapt the cpp and py files to your robot and task.

You can run all the steps below automatically by calling the `demo_robot.bash` script in this directory.

In the task considered in this tutorial, the robot has to throw a ball into a certain area, as illustrated below. The "robot" makes an elliptical movement with its end-effector (blue trajectory), releases the ball (black circles) after 0.6 seconds, so that the ball flies through the air (green trajectory) until it hits the ground. The aim is to throw the ball to a particular position (the green marker on the "floor"). There is a margin of error, illustrated by the dent in the floor.

![alt text](images/task_throw_ball.png  "Illustration of the ball throwing task.")


## Step 1: Train the DMP with a demonstration

It is common practice to initialize a DMP with a demonstrated trajectory, so that the optimization does not have to start from scratch. Given that the optimization algorithms are local, such an initialization is essential to avoid local minima that do not solve the task.

Training is done with the following command:

```
python3 step1_train_dmp_from_trajectory_file.py trajectory.txt results/training --n 15 --save
```

This reads a trajectory from a file (for the format see the Trajectory class), and trains DMP with a different number of basis functions (3 to 15). For each number, a different json file with the DMP is stored. To analyze the fitting process and tune training meta-parameters, the results are written to `results/training/`. Below, the result of training with 5, 10 and 15 basis functions are shown:

![](images/training/trajectory_comparison_5.png  "Results of function approximation with 5 basis functions.")

![](images/training/trajectory_comparison_10.png  "Results of function approximation with 10 basis functions.")

![](images/training/trajectory_comparison_15.png  "Results of function approximation with 15 basis functions.")

In the results above, the fitting with 5 basis functions is not very good, i.e. the reproduced trajectory does not fit the demonstrated trajectory well. This is because 5 basis function do not suffice to fit the data accurately. The fit with 10 basis functions is quite good. Note that with 20 basis functions, the fit is even better. But more basis functions means a higher-dimensional search space for the subsequent optimization, and therefore slower convergence to the (local) optimum. This is the trade-off that needs to be anticipated when choosing the number of basis functions.

In the `results/training` directory, there is also an image that plots the mean absolute error between the demonstration and the reproduction against the number of basis functions. Here we see that 10 is indeed a good choice, and a higher number of basis functions does not lead to higher fitting accuracy.

![](images/training/mean_absolute_errors.png  "Mean absolute error between the demonstration and the reproduction against the number of basis functions.")

Thus, we set the initial DMP for optimization to be the DMP with 10 basis functions:

```
cp results/training/dmp_trained_10.json results/dmp_initial.json
```


## Step 2: Define the task (i.e. cost function) and implement executing rollouts on the robot

Defining the task requires you to make a class that inherits from `dmp_bbo.Task`, and implements the following functions:

* `evaluate_rollout(cost_vars, sample)`. This is the cost function, which takes the cost-relevant variables (`cost_vars`) as an input (and the sample for regularization), and returns the cost associated with the rollout. `cost_vars` thus defines the variables the robot needs to record when performing a rollout, as these variables are required to compute the cost.
* `plot_rollout(cost_vars)`. This function visualizes one rollout.

The task, in this case `TaskThrowBall` is save to file as JSON with the following script:

```
python3 step2_define_task.py results/
```
    
The task converts cost-relevant variables into a cost. The robot, which is responsible for executing the rollouts, should write the cost-relevant variable to a file. Therefore, the user must write an interface to the robot that reads a dmp, executes it, and writes the results to a file containing the cost-relevant variables. In this demo, this interface is the executable `robotExecuteDmp` (compiled from `robotExecuteDmp.cpp`). Executing and plotting the initial DMP can be done with:

```
# Execute the DMP on your robot, and write cost-vars
../../bin/robotExecuteDmp results/training/dmp_trained_10_for_cpp.json tmp_cost_vars.tx
# Plot the cost-vars (which required knowledge of the task)
python3 plot_rollouts.py tmp_cost_vars.txt results/task.json
```
Note that there are often two JSON versions of the DMP, e.g. `dmp_trained_10.json` and `dmp_trained_10_for_cpp.json`. The former is written with `jsonpickle` (which makes it easier to read into Python with `jsonpickle`) and the latter is a simpler custom JSON format (which is easier to read into C++). See `dmpbbo/dmbbo/json_for_cpp.py` for details. 

For the cost function of this task, the only relevant variables are the landing position of the ball, and the accelerations at each time step (accelerations are also penalized). In practice however, I usually store more information in cost_vars for visualization purposes, e.g. the end-effector trajectory and the ball trajectory. These are not needed to compute the cost with `evaluate_rollout(cost_vars, sample)`, but certainly help to provide sensible plots with `plot_rollout(cost_vars)`

Gathering the information for cost-vars can be non-trivial in practice. For instance, for the ball-in-cup experiments on the Meka robot, the end-effector position was recorded by the robot, and stored at each time step. The ball trajectory was recorded with an external camera, passed to the robot, which stored it inside the `cost_vars` matrix alongside the end-effector positions. All of these aspect have been simulated in `robotExecuteDmp.cpp`.

## Step 3: Tune the exploration noise for the optimization

During the stochastic optimization, the parameters of the DMP will be sampled from a Gaussian distribution. The mean of this distribution will be the parameters that resulted from training the DMP with a demonstration through supervised learning.

In this demo, we tune the weights of the radial basis function networks in the DMP, which has been set with `dmp.set_selected_param_names("weights")` in `step1_train_dmp_from_trajectory_file.py`. The function `set_selected_param_names` is part of the `Parameterizable` class from which `Dmp` inherits.

The covariance matrix of the sampling distributions determines the magnitude of exploration. This magnitude is defined in terms of sigma, where the diagonal of the covariance matrix is initialized with sigma^2. Sigma should not be too low, otherwise the stochasticity of the exploration may be smaller than that of the robot movement itself, and no learning can take place. It should also not be too high for safety reasons; your robot may reach acceleration limits, joint limits, or unexpectedly bump into the environment. 

You can tune this parameter by calling the following:

```
# Generate some samples, based on the initial DMP
python3 step3_tune_exploration.py results/dmp_initial.json results/tune_exploration --save --n 10 --sigma   1.0

# Call robotExecuteDmp for each sample
for i_sample in $(seq -f "%02g" 0 9)
do # Run the sampled DMPs on the robot
  ../../bin/robotExecuteDmp results/tune_exploration/sigma_1.000/${i_sample}_dmp_for_cpp.json results/tune_exploration/sigma_1.000/${i_sample}_cost_vars.txt
done
python3 plot_rollouts.py results/tune_exploration/sigma_1.000 results/task.json --save
```

Below the results of exploring with sigma 1.0, 20.0, and 40.0.

![](images/tune_exploration/sigma_1.000/plot_rollouts.png  "Resulting rollouts with sigma = 1.0")

![](images/tune_exploration/sigma_20.000/plot_rollouts.png  "Resulting rollouts with sigma = 20.0")

![](images/tune_exploration/sigma_40.000/plot_rollouts.png  "Resulting rollouts with sigma = 40.0")


The value 1.0 is probably too low, because there is hardly any variation in the end-effector movement. 40.0 is definitely too high! If you execute this on your robot you are a braver person than I (Quote from the license: "This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY". If your bravery breaks your robot, don't blame me!). Given these results, I'd be comfortable with a value between 1.0 and 20.0. We'll continue with 20.0 in this tutorial, as we can't break any robots in simulation. We copy the selected distribution to be the initial one:

```
cp results/tune_exploration/sigma_20.000/distribution.json results/distribution_initial.json
```

If you are performing the optimization with covariance matrix adapation (CMA), i.e. with the UpdaterCovarAdaptation, I would set the initial sigma to 10.0, max_level to 20.0 (so that the exploration is not adapted to more than 20.0) and min_level 1.0 (to avoid premature convergence along one of the dimensions). These parameters are set in `step4_prepare_optimization.py`, which we turn to next.

## Step 4: Prepare the optimization

Whereas Step 1 has defined the search space (with `dmp.set_selected_param_names("weights")`), and Step 3 has determined the initial distribution for the optimization, Step 4 defines how the distribution is updated over time. It does so by initializing an `Updater`, e.g. `UpdaterCovarDecay` or `UpdaterCovarAdaptation`.

Also, it calls `run_optimization_task_prepare`, which sets up various directories, and does a first batch of samples for the optimization process in Step 5. 


## Step 5: Run the optimization step-by-step

Now we have trained a dmp (stored in `dmp_initial.json`), specified the task (stored in `task.json`), and tuned the exploration (stored in `distribution_initial.json`). Now it's time to run the optimization! This is an iterative process with two main steps (and an optional step of plotting intermediate results). Each iteration is called an "update", as it involves one update of the policy parameters.

### Step 5A: Update parameters 

This is a highly automized process, which is called as follows

    python3 step4A_oneOptimizationUpdate.py  results/

This will automatically find the most recent update (e.g. `results/update0083/`) and read all cost_vars in the rollouts in this update directory (which are stored in `update0083/rollout001/cost_vars.txt`, `update0083/rollout002/cost_vars.txt`, etc.). It then computes the costs from each cost_vars (with `task.evaluateRollout(...)`), and updates the policy parameters. Finally, it samples new policy parameters, and saves them in a new update directory (i.e. `update0084/rollout001/policy_parameters.txt`, `update0084/rollout002/policy_parameters.txt`, etc.)

Note: on the first call this script only writes the samples, but does not read the rollouts, as there are none yet.

### Step 5B: Perform rollouts

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

    python3 plot_optimization.py results/

This will automatically determine what the last update directory is, and plot the optimization process so far, as shown below. The left graph shows the evaluation rollout after each update, the red one being the first, and more green rollouts corresponding to more recent rollouts. The second plot shows 2 dimensions of the search space (in this case 2*10 basis functions is 20D). The third plot shows the exploration magnitude (sigma) at each update. Here it decays, with a decay factor of 0.9, which is specified in `step4A_oneOptimizationUpdate.py`. The final graph shows the learning curve. The black line corresponds to the cost of the evaluation rollout, which is based on the updated mean of the Gaussian distribution. The thinner lines correspond to the different cost components, in this case the distance to the landing site, and the cost for accelerations. Finally, the grey dots correspond to each rollout during the optimization, i.e. those sampled from the Gaussian distribution.

![alt text](images/optimization.png  "Optimization results after 15 updates.")

We see that after 15 rollouts, the "robot" has learned to throw the ball in the specified area. The accelerations have increased slightly because the movement to do this requires slightly higher velocities than those in the demonstration.
