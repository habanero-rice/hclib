#!/bin/bash

#SBATCH --job-name=HCLIB-PERFORMANCE-REGRESSION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000m
#SBATCH --time=00:30:00
#SBATCH --mail-user=jmg3@rice.edu
#SBATCH --mail-type=ALL
#SBATCH --export=ALL
#SBATCH --partition=commons

set -e

RUNNING_UNDER_SLURM=1
if [[ -z "$SLURM_JOB_ID" ]]; then
    echo Not executing under SLURM
    RUNNING_UNDER_SLURM=0
fi

if [[ $RUNNING_UNDER_SLURM == 1 ]]; then
    cd $SLURM_SUBMIT_DIR
fi

make clean
make -j

for FOLDER in $(ls rodinia/); do
    if [[ -d rodinia/$FOLDER ]]; then
        cd rodinia/$FOLDER && make clean && make -j && cd ../../
    fi
done

MEDIAN_PY=../../../tools/median.py
MEAN_PY=../../../tools/mean.py
BENCHMARKS=('cilksort 100000000', 'FFT 16384', 'fib 45', 'fib-ddt 45', \
        'nqueens 14', 'qsort 100000000', 'rodinia/backprop/backprop 4194304', \
        'rodinia/bfs/bfs rodinia/bfs/graph1MW_6.txt', \
        'rodinia/b+tree/b+tree.out core 2 file rodinia/b+tree/mil.txt command rodinia/b+tree/command.txt', \
        'rodinia/cfd/euler3d_cpu_double rodinia/cfd/fvcorr.domn.193K',
        'rodinia/heartwall/heartwall rodinia/heartwall/test.avi 20 4')
NTRIALS=10

TIMESTAMP=$(date +%s)
MACHINE=$(hostname -d)
PATH=.:${PATH}

mkdir -p regression-logs-$MACHINE
if [[ $RUNNING_UNDER_SLURM == 1 ]]; then
    LOG_FILE=regression-logs-$MACHINE/$TIMESTAMP.dat
else
    LOG_FILE=regression-logs-$MACHINE/$TIMESTAMP.ignore
fi
REFERENCE_LOG_FILE=$(ls -lrt regression-logs-$MACHINE/ | grep dat | tail -n 1 | awk '{ print $9 }') 
touch $LOG_FILE

for TEST in "${BENCHMARKS[@]}"; do
    TESTNAME=$(basename $(echo $TEST | awk '{ print $1 }'))
    echo Running $TESTNAME

    T=$(for TRIAL in $(seq 1 $NTRIALS); do
        HCLIB_PROFILE_LAUNCH_BODY=1 $TEST 2>&1 | grep 'HCLIB TIME' | awk '{ print $3 }'
    done | python $MEAN_PY)
    echo $TESTNAME $T >> $LOG_FILE
done

if [[ -z "$REFERENCE_LOG_FILE" || ! -f regression-logs-$MACHINE/$REFERENCE_LOG_FILE ]]; then
    echo No available reference peformance information
    exit 1
fi

while read LINE; do
    BENCHMARK=$(basename $(echo $LINE | awk '{ print $1 }'))
    NEW_T=$(echo $LINE | awk '{ print $2 }')
    OLD_T=$(cat regression-logs-$MACHINE/$REFERENCE_LOG_FILE | grep "^$BENCHMARK " | awk '{ print $2 }')
    if [[ -z "$OLD_T" ]]; then
        echo Unable to find an older run of \'$BENCHMARK\' to compare against
    else
        NEW_SPEEDUP=$(echo $OLD_T / $NEW_T | bc -l)
        echo $BENCHMARK $NEW_SPEEDUP
    fi
done < $LOG_FILE
