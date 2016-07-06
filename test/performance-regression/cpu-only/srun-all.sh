#!/bin/bash

#SBATCH --job-name=HCLIB-PERFORMANCE-REGRESSION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000m
#SBATCH --time=08:00:00
#SBATCH --mail-user=jmg3@rice.edu
#SBATCH --mail-type=ALL
#SBATCH --export=ALL
#SBATCH --partition=commons
#SBATCH --exclusive

# Setting the environment variables HCLIB_PERF_CXX and HCLIB_PERF_CC will
# control which compilers are used for these tests. By default the Intel
# compilers are used.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source datasets.sh

MEDIAN_PY=../../../tools/median.py
MEAN_PY=../../../tools/mean.py
STD_PY=../../../tools/std.py

if [[ -z "$RODINIA_DATA_DIR" ]]; then
    echo RODINIA_DATA_DIR must be set to the data directory of the Rodinia benchmark suite
    exit 1
fi

if [[ ! -d $RODINIA_DATA_DIR ]]; then
    setup_data_directories
fi

if [[ -z "$BOTS_ROOT" ]]; then
    echo BOTS_ROOT must be set to the root directory of the BOTS benchmark suite
    exit 1
fi

if [[ -z "$NTRIALS" ]]; then
    NTRIALS=5
fi

if [[ $# -ge 1 ]]; then
    NTRIALS=$1
fi

if [[ $# -ge 2 ]]; then
    TARGET_TEST=$2
fi

RUNNING_UNDER_SLURM=1
if [[ -z "$SLURM_JOB_ID" ]]; then
    echo Not executing under SLURM
    RUNNING_UNDER_SLURM=0
else
    echo Running under SLURM, cd-ing back to $SLURM_SUBMIT_DIR
    cd $SLURM_SUBMIT_DIR
fi

TIMESTAMP=$(date +%s)
set +e
MACHINE=$(hostname -d)
if [[ -z "$MACHINE" ]]; then
    MACHINE=$(hostname)
fi
set -e
PATH=.:${PATH}

mkdir -p regression-logs-$MACHINE
LOG_FILE=regression-logs-$MACHINE/$TIMESTAMP.dat
REFERENCE_LOG_FILE=$(ls -lrt regression-logs-$MACHINE/ | grep dat | tail -n 1 | awk '{ print $9 }') 
touch $LOG_FILE

mkdir -p srun_test_logs

set +e
command -v module >/dev/null
NO_MODULE_COMMAND=$?
command -v srun >/dev/null
NO_SRUN_COMMAND=$?
set -e

if [[ $NO_MODULE_COMMAND -eq 0 ]]; then
    module purge
    source ~/.bash_profile
fi

export LD_LIBRARY_PATH=/opt/apps/software/Core/icc/2015.2.164/composer_xe_2015.2.164/tbb/lib/intel64/gcc4.4:$LD_LIBRARY_PATH

for TEST in "${!DATASETS[@]}"; do
    # Run both flat and recursive versions of the HCLIB benchmarks. Run both
    # tied and untied versions of the OMP benchmarks.
    TEST_EXE=$(echo $TEST | awk -F ',' '{ print $1 }')
    TEST_SIZE=$(echo $TEST | awk -F ',' '{ print $2 }')
    REF_EXE=$TEST_EXE.ref

    if [[ ! -z "$TARGET_TEST" && $TEST_EXE != $TARGET_TEST ]]; then
        echo "Skipping $TEST_EXE"
        continue
    fi

    TEST_ARGS=${DATASETS[$TEST]}

    TESTNAME=$(basename $TEST_EXE)

    TEST_FLAT_LOG=srun_test_logs/tmp.$TESTNAME.$TEST_SIZE.flat.log
    TEST_RECURSIVE_LOG=srun_test_logs/tmp.$TESTNAME.$TEST_SIZE.recursive.log
    REF_TIED_LOG=srun_test_logs/tmp.$TESTNAME.$TEST_SIZE.ref.tied.log
    REF_UNTIED_LOG=srun_test_logs/tmp.$TESTNAME.$TEST_SIZE.ref.untied.log

    echo "Running $TESTNAME ($TEST_SIZE) from $(pwd)"

    EXEC_CMD=""

    if [[ $NO_SRUN_COMMAND -eq 1 ]]; then
        $SCRIPT_DIR/srun_one.sh $NTRIALS "$TEST_ARGS" \
                              $SCRIPT_DIR/$TEST_EXE.flat $SCRIPT_DIR/$TEST_FLAT_LOG \
                              $SCRIPT_DIR/$TEST_EXE.recursive $SCRIPT_DIR/$TEST_RECURSIVE_LOG \
                              $SCRIPT_DIR/$REF_EXE.tied $SCRIPT_DIR/$REF_TIED_LOG \
                              $SCRIPT_DIR/$REF_EXE.untied $SCRIPT_DIR/$REF_UNTIED_LOG
    else
        srun -N 1 --cpus-per-task=12 --exclusive --mem=48000 -p commons -D /tmp \
                                  --time=01:00:00 $SCRIPT_DIR/srun_one.sh $NTRIALS "$TEST_ARGS" \
                                  $SCRIPT_DIR/$TEST_EXE.flat $SCRIPT_DIR/$TEST_FLAT_LOG \
                                  $SCRIPT_DIR/$TEST_EXE.recursive $SCRIPT_DIR/$TEST_RECURSIVE_LOG \
                                  $SCRIPT_DIR/$REF_EXE.tied $SCRIPT_DIR/$REF_TIED_LOG \
                                  $SCRIPT_DIR/$REF_EXE.untied $SCRIPT_DIR/$REF_UNTIED_LOG &
    fi
done

wait

for TEST in "${!DATASETS[@]}"; do
    # Run both flat and recursive versions of the HCLIB benchmarks. Run both
    # tied and untied versions of the OMP benchmarks.
    TEST_EXE=$(echo $TEST | awk -F ',' '{ print $1 }')
    TEST_SIZE=$(echo $TEST | awk -F ',' '{ print $2 }')
    REF_EXE=$TEST_EXE.ref

    if [[ ! -z "$TARGET_TEST" && $TEST_EXE != $TARGET_TEST ]]; then
        echo "Skipping $TEST_EXE"
        continue
    fi

    TEST_ARGS=${DATASETS[$TEST]}

    TESTNAME=$(basename $TEST_EXE)

    TEST_FLAT_LOG=srun_test_logs/tmp.$TESTNAME.$TEST_SIZE.flat.log
    TEST_RECURSIVE_LOG=srun_test_logs/tmp.$TESTNAME.$TEST_SIZE.recursive.log
    REF_TIED_LOG=srun_test_logs/tmp.$TESTNAME.$TEST_SIZE.ref.tied.log
    REF_UNTIED_LOG=srun_test_logs/tmp.$TESTNAME.$TEST_SIZE.ref.untied.log

    echo Parsing $TEST_FLAT_LOG
    FLAT_MEAN=$(cat $TEST_FLAT_LOG | grep 'HCLIB TIME' | awk '{ print $3 }' | python $MEAN_PY)
    FLAT_STD=$(cat $TEST_FLAT_LOG | grep 'HCLIB TIME' | awk '{ print $3 }' | python $STD_PY)

    echo Parsing $TEST_RECURSIVE_LOG
    RECURSIVE_MEAN=$(cat $TEST_RECURSIVE_LOG | grep 'HCLIB TIME' | awk '{ print $3 }' | python $MEAN_PY)
    RECURSIVE_STD=$(cat $TEST_RECURSIVE_LOG | grep 'HCLIB TIME' | awk '{ print $3 }' | python $STD_PY)

    echo Parsing $REF_TIED_LOG
    TIED_MEAN=$(cat $REF_TIED_LOG | grep 'HCLIB TIME' | awk '{ print $3 }' | python $MEAN_PY)
    TIED_STD=$(cat $REF_TIED_LOG | grep 'HCLIB TIME' | awk '{ print $3 }' | python $STD_PY)

    echo Parsing $REF_UNTIED_LOG
    UNTIED_MEAN=$(cat $REF_UNTIED_LOG | grep 'HCLIB TIME' | awk '{ print $3 }' | python $MEAN_PY)
    UNTIED_STD=$(cat $REF_UNTIED_LOG | grep 'HCLIB TIME' | awk '{ print $3 }' | python $STD_PY)

    echo $TESTNAME,$TEST_SIZE $FLAT_MEAN $FLAT_STD $RECURSIVE_MEAN $RECURSIVE_STD $TIED_MEAN $TIED_STD $UNTIED_MEAN $UNTIED_STD >> $LOG_FILE
done

if [[ -z "$REFERENCE_LOG_FILE" || ! -f regression-logs-$MACHINE/$REFERENCE_LOG_FILE ]]; then
    echo No available reference peformance information
    exit 1
fi

while read LINE; do
    BENCHMARK=$(basename $(echo $LINE | awk '{ print $1 }'))
    NEW_FLAT_T=$(echo $LINE | awk '{ print $2 }')
    NEW_RECURSIVE_T=$(echo $LINE | awk '{ print $4 }')
    OLD_FLAT_T=$(cat regression-logs-$MACHINE/$REFERENCE_LOG_FILE | grep "^$BENCHMARK " | awk '{ print $2 }')
    OLD_RECURSIVE_T=$(cat regression-logs-$MACHINE/$REFERENCE_LOG_FILE | grep "^$BENCHMARK " | awk '{ print $4 }')

    if [[ -z "$OLD_FLAT_T" ]]; then
        echo Unable to find an older run of \'$BENCHMARK\' to compare against
    else
        NEW_FLAT_SPEEDUP=$(echo $OLD_FLAT_T / $NEW_FLAT_T | bc -l)
        NEW_RECURSIVE_SPEEDUP=$(echo $OLD_RECURSIVE_T / $NEW_RECURSIVE_T | bc -l)
        echo $BENCHMARK : flat ${NEW_FLAT_SPEEDUP}x, recursive ${NEW_RECURSIVE_SPEEDUP}x
    fi
done < $LOG_FILE

echo

while read LINE; do
    BENCHMARK=$(basename $(echo $LINE | awk '{ print $1 }'))
    NEW_FLAT_T=$(echo $LINE | awk '{ print $2 }')
    NEW_RECURSIVE_T=$(echo $LINE | awk '{ print $4 }')
    OLD_FLAT_T=$(cat regression-logs-$MACHINE/$REFERENCE_LOG_FILE | grep "^$BENCHMARK " | awk '{ print $2 }')
    OLD_RECURSIVE_T=$(cat regression-logs-$MACHINE/$REFERENCE_LOG_FILE | grep "^$BENCHMARK " | awk '{ print $4 }')

    if [[ -z "$OLD_FLAT_T" ]]; then
        echo Unable to find an older run of \'$BENCHMARK\' to compare against
    else
        NEW_FLAT_SPEEDUP=$(echo $OLD_FLAT_T / $NEW_FLAT_T - 1.0 | bc -l)
        NEW_RECURSIVE_SPEEDUP=$(echo $OLD_RECURSIVE_T / $NEW_RECURSIVE_T - 1.0 | bc -l)
        NORMALIZED_FLAT_SPEEDUP=${NEW_FLAT_SPEEDUP#-}
        NORMALIZED_RECURSIVE_SPEEDUP=${NEW_RECURSIVE_SPEEDUP#-}

        FLAT_ABOVE_THRESHOLD=$(echo $NORMALIZED_FLAT_SPEEDUP '>' 0.05 | bc -l)
        RECURSIVE_ABOVE_THRESHOLD=$(echo $NORMALIZED_RECURSIVE_SPEEDUP '>' 0.05 | bc -l)

        if [[ $FLAT_ABOVE_THRESHOLD -eq 1 || $RECURSIVE_ABOVE_THRESHOLD -eq 1 ]]; then
            echo WARNING $BENCHMARK : flat ${NEW_FLAT_SPEEDUP}x, recursive ${NEW_RECURSIVE_SPEEDUP}x
        fi

    fi
done < $LOG_FILE
