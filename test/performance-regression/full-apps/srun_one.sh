#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function run_test() {
    EXE=$1
    LOG=$2
    TEST_ARGS=$3
    NTRIALS=$4

    export LD_LIBRARY_PATH=/opt/apps/software/Core/icc/2015.2.164/composer_xe_2015.2.164/tbb/lib/intel64/gcc4.4:$LD_LIBRARY_PATH

    for TRIAL in $(seq 1 $NTRIALS); do
        LD_PRELOAD=libtbbmalloc_proxy.so.2 HCLIB_PROFILE_LAUNCH_BODY=1 taskset --cpu-list 0,1,2,3,4,5,6,7,8,9,10,11 $EXE $TEST_ARGS 2>&1
    done > $LOG
}

NTRIALS=$1
TEST_ARGS=$2

FLAT_EXE=$3
FLAT_LOG=$4

RECURSIVE_EXE=$5
RECURSIVE_LOG=$6

TIED_EXE=$7
TIED_LOG=$8

UNTIED_EXE=$9
UNTIED_LOG=${10}

for EXE in $FLAT_EXE $RECURSIVE_EXE $TIED_EXE $UNTIED_EXE; do
    if [[ ! -f $EXE ]]; then
        echo Missing executable $EXE
        exit 1
    fi
done

run_test $FLAT_EXE $FLAT_LOG "$TEST_ARGS" $NTRIALS
run_test $RECURSIVE_EXE $RECURSIVE_LOG "$TEST_ARGS" $NTRIALS
run_test $TIED_EXE $TIED_LOG "$TEST_ARGS" $NTRIALS
run_test $UNTIED_EXE $UNTIED_LOG "$TEST_ARGS" $NTRIALS
