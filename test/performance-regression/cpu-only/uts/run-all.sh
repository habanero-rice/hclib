#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

NODES="1 2 4 8"

if [[ $# -lt 1 ]]; then
    echo 'usage: run-all.sh exe [nodes]'
    exit 1
fi

EXE=$1
if [[ $# -ge 2 ]]; then
    NODES=$2
fi

if [[ "$SCRIPT_DIR" != "$(pwd)" ]]; then
    echo This script must be run from the UTS directory
    exit 1
fi

rm -f $SCRIPT_DIR/$EXE
make $EXE
echo

source sample_trees.sh

for DATASET in T1L; do
    for NODE in $NODES; do
        echo $EXE $DATASET $NODE
        SBATCH_FILE=slurm/tmp/$EXE.$DATASET.$NODE.sh
        sed -e "s/N_NODES/$NODE/g" slurm/template.sh > $SBATCH_FILE
        sed -i -e "s/STDOUT_FILE/slurm\/output\/$EXE.$DATASET.$NODE.stdout/g" $SBATCH_FILE
        sed -i -e "s/STDERR_FILE/slurm\/output\/$EXE.$DATASET.$NODE.stderr/g" $SBATCH_FILE
        sed -i -e "s|UTS_DIR|$SCRIPT_DIR|g" $SBATCH_FILE
        sed -i -e "s|UTS_EXE|$EXE|g" $SBATCH_FILE
        sed -i -e "s|TREE_ARGS|$(printenv $DATASET)|g" $SBATCH_FILE
        sbatch $SBATCH_FILE
    done
done
