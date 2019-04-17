#!/bin/bash

# Train a dmp with a trajectory, and save it to a file
./trainDmpFromTrajectoryFile trajectory.txt results/dmp.xml results/policy_parameters.txt results/train/ 6

# Plot the results of training the dmp
python3 trainDmpFromTrajectoryFilePlot.py results/train/

# Generate some samples to tune the exploration, run the dmp, and plot the results
python3 tuneExploration.py results/dmp.xml results/policy_parameters.txt results/tune_exploration/ 10.5 10

