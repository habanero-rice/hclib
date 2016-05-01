#!/bin/bash

#SBATCH --job-name=GUPS-N_NODESnodes-PROC_PER_NODEpernode-SET_OMPthreads
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
#SBATCH --mail-user=jmg3@rice.edu
#SBATCH --mail-type=ALL

# source ~/.bash_profile

set -e

cd GUPS_DIR

# LD_LIBRARY_PATH=/scratch/jmg3/staging/openmpi-install/lib:$LD_LIBRARY_PATH HCLIB_WORKERS=SET_OMP OMP_NUM_THREADS=SET_OMP srun ./GUPS_EXE GUPS_ARGS
HCLIB_WORKERS=SET_OMP OMP_NUM_THREADS=SET_OMP srun ./GUPS_EXE GUPS_ARGS
