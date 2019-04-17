#!/bin/bash

# Train a dmp with a trajectory, and save it to a file
./trainDmpFromTrajectoryFile trajectory.txt results dmp.xml 6

# Plot the results of training the dmp
python3 trainDmpFromTrajectoryFilePlot.py results/

# Generate some samples to tune the exploration
python3 tuneExploration.py results/ 0.1 10

# Perform rolouts with these samples.txt
./performDmpRollouts results/dmp.xml results/tune_exploration/