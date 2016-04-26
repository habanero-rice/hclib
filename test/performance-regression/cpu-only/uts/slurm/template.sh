#!/bin/bash

#SBATCH --job-name=UTS
#SBATCH --partition=commons
#SBATCH --time=TIME_LIMIT
#SBATCH --nodes=N_NODES
#SBATCH --exclusive
#SBATCH -o STDOUT_FILE
#SBATCH -e STDERR_FILE
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=PROC_PER_NODE
#SBATCH --cpus-per-task=CPUS_PER_PROC
#SBATCH --mem=MaxMemPerNode

source ~/.bash_profile

set -e

cd UTS_DIR

HCLIB_WORKERS=SET_OMP OMP_NUM_THREADS=SET_OMP srun ./UTS_EXE TREE_ARGS
# GASNET_BACKTRACE=1 MPIRUN_CMD="srun %C" MPIRUN_CMD_OK=1 OMP_NUM_THREADS=SET_OMP oshrun -n N_NODES -N N_NODES ./UTS_EXE TREE_ARGS
