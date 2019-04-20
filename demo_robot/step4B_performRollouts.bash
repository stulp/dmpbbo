if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <main directory>"
    echo "Example: $0 results/"
    exit -1
fi

DIRECTORY=$1

# Get the most recent update directory
# `tail -n 1` because we only need the last (most recent) one
UPDATE_DIR=`find ${DIRECTORY} -type d -name 'update*' | sort | tail -n 1`
echo "bash   | Update directory is ${UPDATE_DIR}"

echo "bash   | Calling ./robotPerformRollouts.bash $DIRECTORY/dmp.xml ${UPDATE_DIR}"

./robotPerformRollouts.bash $DIRECTORY/dmp.xml ${UPDATE_DIR}