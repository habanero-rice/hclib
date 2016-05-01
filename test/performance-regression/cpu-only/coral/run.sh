#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

NCORES=$(cat /proc/cpuinfo | grep processor | wc -l)

N_OMP_CORES=$NCORES
PROCS_PER_NODE=1
TIME_LIMIT="00:05:00"

declare -A ARGS_MAP
ARGS_MAP[kmi_hash]='-n 500000 -m 4000000'

if [[ $# -lt 2 ]]; then
    echo 'usage: run-all.sh exe nodes [omp-cores] [procs-per-node] [time-limit]'
    exit 1
fi

EXE=$1
NODES=$2
if [[ $# -ge 3 ]]; then
    N_OMP_CORES=$3
fi
if [[ $# -ge 4 ]]; then
    PROCS_PER_NODE=$4
fi
if [[ $# -ge 5 ]]; then
    TIME_LIMIT=$5
fi

EXE_PATH=$(find . -samefile $EXE)
EXE_DIR=$(echo $EXE_PATH | cut -d "/" -f2)
EXE_NAME=$(basename $EXE_PATH)
if [[ -z "${ARGS_MAP[$EXE_DIR]}" ]]; then
    echo "EXE_DIR=$EXE_DIR is unknown, cannot get arguments for that benchmark"
    exit 1
fi
EXE_ARGS=${ARGS_MAP[$EXE_DIR]}

if [[ "$SCRIPT_DIR" != "$(pwd)" ]]; then
    echo This script must be run from the coral directory
    exit 1
fi

for NODE in $NODES; do
    for PER_NODE in $PROCS_PER_NODE; do

        REMAINDER=$(($NCORES%$PER_NODE))
        if [[ $REMAINDER -ne 0 ]]; then
            echo "Invalid per-node ($PER_NODE) for $NCORES cors"
            exit 1
        fi
        PER_PROC=$(($NCORES/$PER_NODE))

        for CORES in $N_OMP_CORES; do
            echo $EXE dataset=$DATASET nodes=$NODE ppn=$PER_NODE omp-cores=$CORES

            RUN_LBL=$EXE_DIR.$EXE_NAME.${NODE}nodes.${PER_NODE}per_node.${CORES}omp_cores
            SBATCH_FILE=slurm/scripts/$RUN_LBL.sh

            sed -e "s|BENCHMARK|$EXE_DIR-$EXE_NAME|g" slurm/template.sh > $SBATCH_FILE
            sed -i -e "s|N_NODES|$NODE|g" $SBATCH_FILE
            sed -i -e "s|PROC_PER_NODE|$PER_NODE|g" $SBATCH_FILE
            sed -i -e "s|SET_OMP|$CORES|g" $SBATCH_FILE
            sed -i -e "s|TIME_LIMIT|$TIME_LIMIT|g" $SBATCH_FILE
            sed -i -e "s|STDOUT_FILE|slurm\/output\/$RUN_LBL.stdout|g" $SBATCH_FILE
            sed -i -e "s|STDERR_FILE|slurm\/output\/$RUN_LBL.stderr|g" $SBATCH_FILE
            sed -i -e "s|CPUS_PER_PROC|$PER_PROC|g" $SBATCH_FILE
            sed -i -e "s|BENCHMARK_DIR|$SCRIPT_DIR|g" $SBATCH_FILE
            sed -i -e "s|BENCHMARK_EXE|$EXE|g" $SBATCH_FILE
            sed -i -e "s|BENCHMARK_ARGS|$EXE_ARGS|g" $SBATCH_FILE
            sbatch $SBATCH_FILE
        done
    done
done
