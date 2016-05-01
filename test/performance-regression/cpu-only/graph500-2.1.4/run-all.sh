#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

NCORES=$(cat /proc/cpuinfo | grep processor | wc -l)

NODES="4"
PROCS_PER_NODE=1

if [[ $# -lt 1 ]]; then
    echo 'usage: run-all.sh exe [nodes]'
    exit 1
fi

EXE=$1
if [[ $# -ge 2 ]]; then
    NODES=$2
fi

if [[ ! -f $SCRIPT_DIR/mpi/$EXE ]]; then
    echo Invalid exe, $SCRIPT_DIR/mpi/$EXE does not seem to be a file
    exit 1
fi

if [[ "$SCRIPT_DIR" != "$(pwd)" ]]; then
    echo This script must be run from the Graph500 directory
    exit 1
fi

for NODE in $NODES; do
    echo $EXE nodes=$NODE

    SBATCH_FILE=slurm/scripts/$EXE.$NODE.sh

    sed -e "s/NNODES/$NODE/g" slurm/template.sh > $SBATCH_FILE
    sed -i -e "s|EXE_NAME|$EXE|g" $SBATCH_FILE
    sbatch $SBATCH_FILE
done
