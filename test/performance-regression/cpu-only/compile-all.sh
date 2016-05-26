#!/bin/bash

#SBATCH --job-name=HCLIB-PERFORMANCE-REGRESSION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000m
#SBATCH --time=03:00:00
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

function compile_all() {
    STYLE=$1
    REF=$2

    MAKE_FILE=Makefile
    if [[ $REF -eq 1 ]]; then
        MAKE_FILE=Makefile.ref
    fi

    if [[ -f custom.mak ]]; then
        unlink custom.mak
    fi
    ln -s custom.$STYLE.mak custom.mak

    for FOLDER in $(ls rodinia/); do
        if [[ -d rodinia/$FOLDER ]]; then
            echo Compiling rodinia/$FOLDER
            cd rodinia/$FOLDER && make -f $MAKE_FILE clean && make -f $MAKE_FILE && cd ../../
        fi
    done

    for FOLDER in $(ls bots/); do
        if [[ -d bots/$FOLDER ]]; then
            echo Compiling bots/$FOLDER
            cd bots/$FOLDER && make -f $MAKE_FILE clean && make -f $MAKE_FILE && cd ../../
        fi
    done

    for FOLDER in $(ls kastors-1.1/); do
        if [[ -d kastors-1.1/$FOLDER && $FOLDER != common ]]; then
            echo Compiling kastor-1.1/$FOLDER
            cd kastors-1.1/$FOLDER && make -f $MAKE_FILE clean && make -f $MAKE_FILE && cd ../../
        fi
    done

    for FIX in 'bots/alignment_single/alignment.ref.icc.omp-tasks bots/alignment_single/alignment.icc.single-omp-tasks.ref' \
            'bots/alignment_for/alignment.icc.for.ref-omp-tasks bots/alignment_for/alignment.icc.for-omp-tasks.ref' \
            'bots/fft/fft.ref.icc.omp-tasks bots/fft/fft.icc.omp-tasks.ref' \
            'bots/fib/fib.ref.icc.omp-tasks bots/fib/fib.icc.omp-tasks.ref' \
            'bots/floorplan/floorplan.ref.icc.omp-tasks bots/floorplan/floorplan.icc.omp-tasks.ref' \
            'bots/health/health.ref.icc.omp-tasks bots/health/health.icc.omp-tasks.ref' \
            'bots/nqueens/nqueens.ref.icc.omp-tasks bots/nqueens/nqueens.icc.omp-tasks.ref' \
            'bots/sort/sort.ref.icc.omp-tasks bots/sort/sort.icc.omp-tasks.ref' \
            'bots/sparselu_for/sparselu.ref.icc.omp-tasks bots/sparselu_for/sparselu.icc.for-omp-tasks.ref' \
            'bots/sparselu_single/sparselu.ref.icc.omp-tasks bots/sparselu_single/sparselu.icc.single-omp-tasks.ref' \
            'bots/strassen/strassen.ref.icc.omp-tasks bots/strassen/strassen.icc.omp-tasks.ref' \
            'bots/uts/uts.ref.icc.omp-tasks bots/uts/uts.icc.omp-tasks.ref'; do
        SRC=$(echo $FIX | awk '{ print $1 }')
        DST=$(echo $FIX | awk '{ print $2 }')

        if [[ -f $SRC ]]; then
            mv $SRC $DST
        fi
    done

    for TEST in "${!DATASETS[@]}"; do
        TEST_EXE=$(echo $TEST | awk -F ',' '{ print $1 }')
        TEST_SIZE=$(echo $TEST | awk -F ',' '{ print $2 }')
        REF_EXE=$TEST_EXE.ref

        SRC_EXE=$TEST_EXE
        DST_EXE=$TEST_EXE.$STYLE
        if [[ $REF -eq 1 ]]; then
            SRC_EXE=$REF_EXE
            DST_EXE=$REF_EXE.$STYLE
        fi


        # Every test must have a 'large' dataset. If we just do this move on
        # every member of DATASETS, we'll run in to problems with duplicate
        # entries in DATASETS because of 'small' and 'large' datasets.
        if [[ $TEST_SIZE == large ]]; then
            mv $SRC_EXE $DST_EXE
        fi
    done

    echo Compilation for $STYLE completed!
}

compile_all tied 1
compile_all untied 1
compile_all flat 0
compile_all recursive 0

for TEST in "${!DATASETS[@]}"; do
    TEST_EXE=$(echo $TEST | awk -F ',' '{ print $1 }')
    REF_EXE=$TEST_EXE.ref

    for EXE in $TEST_EXE.flat $TEST_EXE.recursive $REF_EXE.tied $REF_EXE.untied; do
        if [[ ! -f $EXE ]]; then
            echo Missing executable $EXE
        fi
    done
done

echo
echo 'Done!'
