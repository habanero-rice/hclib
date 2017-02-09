#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

NCORES=$(cat /proc/cpuinfo | grep processor | wc -l)

NODES="1 2 4 8"
N_OMP_CORES=$NCORES
PROCS_PER_NODE=1
TIME_LIMIT="00:02:00"

if [[ $# -lt 1 ]]; then
    echo 'usage: run-all.sh exe [nodes] [omp-cores] [procs-per-node] [time-limit]'
    exit 1
fi

EXE=$1
if [[ $# -ge 2 ]]; then
    NODES=$2
fi
if [[ $# -ge 3 ]]; then
    N_OMP_CORES=$3
fi
if [[ $# -ge 4 ]]; then
    PROCS_PER_NODE=$4
fi
if [[ $# -ge 5 ]]; then
    TIME_LIMIT=$5
fi

if [[ "$SCRIPT_DIR" != "$(pwd)" ]]; then
    echo This script must be run from the UTS directory
    exit 1
fi

if [[ ! -f $SCRIPT_DIR/slurm/slurm.conf ]]; then
    echo You must create a file at $SCRIPT_DIR/slurm/slurm.conf.
    echo See $SCRIPT_DIR/slurm/slurm.conf.example for an example
    exit 1
fi

source $SCRIPT_DIR/slurm/slurm.conf

rm -f $SCRIPT_DIR/$EXE
make $EXE
echo

source sample_trees.sh

for DATASET in T1L; do
# for DATASET in T1XL; do
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

                SBATCH_FILE=slurm/tmp/$EXE.$DATASET.$NODE.$CORES.sh
                sed -e "s/N_NODES/$NODE/g" slurm/template.sh > $SBATCH_FILE
                sed -i -e "s/STDOUT_FILE/slurm\/output\/$EXE.$DATASET.$NODE.$PER_NODE.$CORES.stdout/g" $SBATCH_FILE
                sed -i -e "s/STDERR_FILE/slurm\/output\/$EXE.$DATASET.$NODE.$PER_NODE.$CORES.stderr/g" $SBATCH_FILE
                sed -i -e "s|UTS_DIR|$SCRIPT_DIR|g" $SBATCH_FILE
                sed -i -e "s|UTS_EXE|$EXE|g" $SBATCH_FILE
                sed -i -e "s|TREE_ARGS|$(printenv $DATASET)|g" $SBATCH_FILE
                sed -i -e "s|SET_OMP|$CORES|g" $SBATCH_FILE
                sed -i -e "s|PROC_PER_NODE|$PER_NODE|g" $SBATCH_FILE
                sed -i -e "s|CPUS_PER_PROC|$PER_PROC|g" $SBATCH_FILE
                sed -i -e "s|TIME_LIMIT|$TIME_LIMIT|g" $SBATCH_FILE
                sed -i -e "s|SLURM_QUEUE|$SLURM_QUEUE|g" $SBATCH_FILE
                sbatch $SBATCH_FILE
            done
        done
    done
done
