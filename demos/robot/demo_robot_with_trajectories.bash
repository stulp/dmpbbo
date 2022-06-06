#!/bin/bash

D=results_with_traj

################################################################################
# STEP 1: Train the DMP with a trajectory. Try it with different # basis functions
python3 step1_train_dmp_from_trajectory_file.py trajectory.txt ${D}/training --n 15 --save
# 10 basis functions look good; choose it as initial DMP for optimization
cp ${D}/training/dmp_trained_10.json ${D}/dmp_initial.json


################################################################################
# STEP 2: Define and save the task
python3 step2_define_task.py ${D} task.json


################################################################################
# STEP 3: Tune the exploration noise

# Low exploration noise
python3 step3_tune_exploration.py ${D}/dmp_initial.json ${D}/tune_exploration --traj --sigma   1.0 
DU="${D}/tune_exploration/sigma_1.000"
for i_sample in $(seq -f "%02g" 0 9)
do # Run the sampled DMPs on the robot
  ../../bin/robotExecuteTrajectory ${DU}/${i_sample}_traj.txt ${DU}/${i_sample}_cost_vars.txt
done
python3 plot_rollouts.py ${DU} ${D}/task.json --save # Save the results as a png

# Medium exploration noise
python3 step3_tune_exploration.py ${D}/dmp_initial.json ${D}/tune_exploration --traj --sigma  20.0
DU="${D}/tune_exploration/sigma_20.000"
for i_sample in $(seq -f "%02g" 0 9)
do # Run the sampled DMPs on the robot
  ../../bin/robotExecuteTrajectory ${DU}/${i_sample}_traj.txt ${DU}/${i_sample}_cost_vars.txt
done
python3 plot_rollouts.py ${DU} ${D}/task.json --save # Save the results as a png

# Low exploration noise
python3 step3_tune_exploration.py ${D}/dmp_initial.json ${D}/tune_exploration --traj --sigma 40.0
DU="${D}/tune_exploration/sigma_40.000"
for i_sample in $(seq -f "%02g" 0 9)
do # Run the sampled DMPs on the robot
  ../../bin/robotExecuteTrajectory ${DU}/${i_sample}_traj.txt ${DU}/${i_sample}_cost_vars.txt
done
python3 plot_rollouts.py ${DU} ${D}/task.json --save # Save the results as a png


# # 10.0 looks good; choose it as initial distribution
cp ${D}/tune_exploration/sigma_20.000/distribution.json ${D}/distribution_initial.json


################################################################################
# STEP 4: Prepare the optimization
python3 step4_prepare_optimization.py ${D} --traj


################################################################################
# STEP 5: Run the optimization
for i_update in $(seq -f "%05g" 0 15)
do
  
  # Run the sampled DMPs on the robot
  DU="${D}/update${i_update}"
  # Evaluation rollout
  ../../bin/robotExecuteTrajectory ${DU}/eval_traj.txt ${DU}/eval_cost_vars.txt
  # Samples rollouts
  for i in $(seq -f "%03g" 0 4)
  do
    ../../bin/robotExecuteTrajectory ${DU}/${i}_traj.txt ${DU}/${i}_cost_vars.txt
  done
  
  python3 step5_one_optimization_update.py ${D} ${i_update} --traj
  
done
  
python3 plot_optimization.py ${D} --save