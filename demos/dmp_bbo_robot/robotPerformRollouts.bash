if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input dmp> <directory with rollouts>"
    exit -1
fi

INPUT_DMP=$1
DIRECTORY=$2

# Get the rollout directory within the update directory
ROLLOUT_DIRS=`find ${DIRECTORY} -type d -name 'rollout*' | sort`
echo "bash   | Rollout directories are ${ROLLOUT_DIRS}"

# Call execute dmp for each rollout
for CUR_DIR in $ROLLOUT_DIRS
do
  echo "bash   | Calling ./robotPerformRollout $INPUT_DMP $CUR_DIR/cost_vars.txt $CUR_DIR/policy_parameters.txt $CUR_DIR/dmp.xml"
  ./robotPerformRollout $INPUT_DMP $CUR_DIR/cost_vars.txt $CUR_DIR/policy_parameters.txt $CUR_DIR/dmp.xml
done