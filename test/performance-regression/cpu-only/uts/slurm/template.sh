#!/bin/bash

#SBATCH --job-name=UTS-N_NODESnodes-PROC_PER_NODEpernode-SET_OMPthreads
#SBATCH --partition=SLURM_QUEUE
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

cd UTS_DIR

# LD_LIBRARY_PATH=/scratch/jmg3/staging/openmpi-install/lib:/opt/apps/software/Compiler/intel/2015.2.164/CUDA/7.0.28/lib64:/opt/apps/software/Compiler/intel/2015.2.164/impi/5.0.3.048/lib64:$LD_LIBRARY_PATH HCLIB_WORKERS=SET_OMP OMP_NUM_THREADS=SET_OMP srun ./UTS_EXE TREE_ARGS
HCLIB_WORKERS=SET_OMP OMP_NUM_THREADS=SET_OMP srun ./UTS_EXE TREE_ARGS
# GASNET_BACKTRACE=1 MPIRUN_CMD="srun %C" MPIRUN_CMD_OK=1 OMP_NUM_THREADS=SET_OMP oshrun -n N_NODES -N N_NODES ./UTS_EXE TREE_ARGS
