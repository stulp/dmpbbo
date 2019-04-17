#!/bin/bash

################################
# STEP 1: Train the DMP with a trajectory

# Train a dmp with a trajectory, and save it to a file
./step1B_trainDmpFromTrajectoryFile trajectory.txt results/dmp.xml results/policy_parameters.txt results/train/ 6

# Plot the results of training the dmp
python3 step1B_trainDmpFromTrajectoryFilePlot.py results/train/


################################
# STEP 2: Define and save the task

python3 step2_defineTask.py results/

################################
# STEP 3: Tune the exploration noise

# Generate some samples to tune the exploration, run the dmp, and plot the results
# The penultimate parameters is the magnitude of the exploration. Here we try three values

python3 step3A_tuneExploration.py results/policy_parameters.txt results/distribution_initial_covar.txt results/tune_exploration_0.1/ 0.1 12
./step3B_performExplorationRollouts.bash results/dmp.xml results/tune_exploration_0.1/
python3 step3C_tuneExplorationPlot.py results/tune_exploration_0.1/ results/task.p

python3 step3A_tuneExploration.py results/policy_parameters.txt results/distribution_initial_covar.txt results/tune_exploration_10.0/ 10.0 12
./step3B_performExplorationRollouts.bash results/dmp.xml results/tune_exploration_10.0/
python3 step3C_tuneExplorationPlot.py results/tune_exploration_10.0/ results/task.p

python3 step3A_tuneExploration.py results/policy_parameters.txt results/distribution_initial_covar.txt results/tune_exploration_1.0/ 1.0 12
./step3B_performExplorationRollouts.bash results/dmp.xml results/tune_exploration_1.0/
python3 step3C_tuneExplorationPlot.py results/tune_exploration_1.0/ results/task.p