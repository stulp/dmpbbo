#!/bin/bash

D=results

# Adding --show to (most) python scripts below will plot the results

# STEP 1: Train the DMP with a trajectory. Try it with different # basis functions
python3 step1_train_dmp_from_trajectory_file.py trajectory.txt ${D}/training --n 15
# 10 basis functions look good; choose it as initial DMP for optimization
cp ${D}/training/dmp_trained_10.json ${D}/dmp_initial.json

# STEP 2: Define and save the task
python3 step2_define_task.py ${D} task.json

# STEP 3: Tune the exploration noise
python3 step3_tune_exploration.py ${D}/dmp_initial.json ${D}/tune_exploration --sigma   1.0
python3 step3_tune_exploration.py ${D}/dmp_initial.json ${D}/tune_exploration --sigma  10.0
python3 step3_tune_exploration.py ${D}/dmp_initial.json ${D}/tune_exploration --sigma 100.0
# 10.0 looks good; choose it as initial distribution
cp ${D}/tune_exploration/sigma_10.000/distribution.json ${D}/distribution_initial.json

# STEP 4: Prepare the optimization
python3 step4_prepare_optimization.py ${D}

# STEP 5: Run the optimization
for i_update in $(seq -f "%05g" 0 15)
do
  
  # Run the sampled DMPs on the robot
  DU="${D}/update${i_update}"
  ../../bin/robotExecuteDmp ${DU}/eval_dmp_for_cpp.json ${DU}/eval_cost_vars.txt
  for i in $(seq -f "%03g" 0 4)
  do
    ../../bin/robotExecuteDmp ${DU}/${i}_dmp_for_cpp.json ${DU}/${i}_cost_vars.txt
  done
  
  python3 step5_one_optimization_update.py ${D} ${i_update}
  
done
  
python3 step6_plot_optimization.py ${D}