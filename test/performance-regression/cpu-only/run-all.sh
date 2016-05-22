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

source datasets.sh

MEDIAN_PY=../../../tools/median.py
MEAN_PY=../../../tools/mean.py
STD_PY=../../../tools/std.py

function run_one_test() {
    EXE=$1
    LOG=$2
    NTRIALS=$3
    TEST_ARGS=$4

    if [[ ! -f $EXE ]]; then
        echo Missing executable $EXE
        exit 1
    fi

    for TRIAL in $(seq 1 $NTRIALS); do
        LD_PRELOAD=libtbbmalloc_proxy.so.2 HCLIB_PROFILE_LAUNCH_BODY=1 $EXE $TEST_ARGS 2>&1
    done > $LOG

    T=$(cat $LOG | grep 'HCLIB TIME' | awk '{ print $3 }' | python $MEAN_PY)
    STD=$(cat $LOG | grep 'HCLIB TIME' | awk '{ print $3 }' | python $STD_PY)

    echo $T $STD
}

if [[ -z "$RODINIA_DATA_DIR" ]]; then
    echo RODINIA_DATA_DIR must be set to the data directory of the Rodinia benchmark suite
    exit 1
fi

if [[ -z "$BOTS_ROOT" ]]; then
    echo BOTS_ROOT must be set to the root directory of the BOTS benchmark suite
    exit 1
fi

if [[ -z "$NTRIALS" ]]; then
    NTRIALS=10
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
if [[ $RUNNING_UNDER_SLURM == 1 ]]; then
    LOG_FILE=regression-logs-$MACHINE/$TIMESTAMP.dat
else
    LOG_FILE=regression-logs-$MACHINE/$TIMESTAMP.ignore
fi
REFERENCE_LOG_FILE=$(ls -lrt regression-logs-$MACHINE/ | grep dat | tail -n 1 | awk '{ print $9 }') 
touch $LOG_FILE

mkdir -p test_logs

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

    TEST_FLAT_LOG=test_logs/tmp.$TESTNAME.$TEST_SIZE.flat.log
    TEST_RECURSIVE_LOG=test_logs/tmp.$TESTNAME.$TEST_SIZE.recursive.log
    REF_TIED_LOG=test_logs/tmp.$TESTNAME.$TEST_SIZE.ref.tied.log
    REF_UNTIED_LOG=test_logs/tmp.$TESTNAME.$TEST_SIZE.ref.untied.log

    echo "Running $TESTNAME ($TEST_SIZE) from $(pwd)"

    # tied doesn't matter here
    FLAT_RESULTS=$(run_one_test $TEST_EXE.flat $TEST_FLAT_LOG $NTRIALS "$TEST_ARGS")
    FLAT_MEAN=$(echo $FLAT_RESULTS | awk '{ print $1 }')
    FLAT_STD=$(echo $FLAT_RESULTS | awk '{ print $2 }')
    RECURSIVE_RESULTS=$(run_one_test $TEST_EXE.recursive $TEST_RECURSIVE_LOG $NTRIALS "$TEST_ARGS")
    RECURSIVE_MEAN=$(echo $RECURSIVE_RESULTS | awk '{ print $1 }')
    RECURSIVE_STD=$(echo $RECURSIVE_RESULTS | awk '{ print $2 }')
    # flat doesn't matter here
    TIED_RESULTS=$(run_one_test $REF_EXE.tied $REF_TIED_LOG $NTRIALS "$TEST_ARGS")
    TIED_MEAN=$(echo $TIED_RESULTS | awk '{ print $1 }')
    TIED_STD=$(echo $TIED_RESULTS | awk '{ print $2 }')
    UNTIED_RESULTS=$(run_one_test $REF_EXE.untied $REF_UNTIED_LOG $NTRIALS "$TEST_ARGS")
    UNTIED_MEAN=$(echo $UNTIED_RESULTS | awk '{ print $1 }')
    UNTIED_STD=$(echo $UNTIED_RESULTS | awk '{ print $2 }')

    echo $TESTNAME,$TEST_SIZE $FLAT_MEAN $FLAT_STD $RECURSIVE_MEAN $RECURSIVE_STD $TIED_MEAN $TIED_STD $UNTIED_MEAN $UNTIED_STD >> $LOG_FILE
done

# if [[ -z "$REFERENCE_LOG_FILE" || ! -f regression-logs-$MACHINE/$REFERENCE_LOG_FILE ]]; then
#     echo No available reference peformance information
#     exit 1
# fi
# 
# while read LINE; do
#     BENCHMARK=$(basename $(echo $LINE | awk '{ print $1 }'))
#     NEW_T=$(echo $LINE | awk '{ print $2 }')
#     OLD_T=$(cat regression-logs-$MACHINE/$REFERENCE_LOG_FILE | grep "^$BENCHMARK " | awk '{ print $2 }')
#     if [[ -z "$OLD_T" ]]; then
#         echo Unable to find an older run of \'$BENCHMARK\' to compare against
#     else
#         NEW_SPEEDUP=$(echo $OLD_T / $NEW_T | bc -l)
#         echo $BENCHMARK $NEW_SPEEDUP
#     fi
# done < $LOG_FILE
