if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input dmp> <directory with rollouts>"
    exit -1
fi

INPUT_DMP=$1
DIRECTORY=$2

./robotPerformRollouts.bash $INPUT_DMP ${DIRECTORY}