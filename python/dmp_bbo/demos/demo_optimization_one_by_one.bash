#!/bin/bash

DIREC="/tmp/demo_optimization_one_by_one"

# Yes, I know there are for loops in bash ;-)
# But this makes it really explicit how to call the scripts

python demo_one_update.py $DIREC/
python demo_perform_rollouts.py $DIREC/update00000 # Replace this with your robot

python demo_one_update.py $DIREC/
python demo_perform_rollouts.py $DIREC/update00001 # Replace this with your robot

python demo_one_update.py $DIREC/
python demo_perform_rollouts.py $DIREC/update00002 # Replace this with your robot

python demo_one_update.py $DIREC/
python demo_perform_rollouts.py $DIREC/update00003 # Replace this with your robot

python demo_one_update.py $DIREC/
python demo_perform_rollouts.py $DIREC/update00004 # Replace this with your robot

python demo_one_update.py $DIREC/ plotresults
python demo_perform_rollouts.py $DIREC/update00005 # Replace this with your robot

python demo_one_update.py $DIREC/ 
python demo_perform_rollouts.py $DIREC/update00006 # Replace this with your robot

python demo_one_update.py $DIREC/ plotresults
python demo_perform_rollouts.py $DIREC/update00007 # Replace this with your robot

python demo_one_update.py $DIREC/ 
python demo_perform_rollouts.py $DIREC/update00008 # Replace this with your robot

python demo_one_update.py $DIREC/ plotresults
