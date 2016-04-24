#!/bin/bash

#SBATCH --job-name=UTS
#SBATCH --partition=commons
#SBATCH --time=00:15:00
#SBATCH --nodes=N_NODES
#SBATCH --exclusive
#SBATCH -o STDOUT_FILE
#SBATCH -e STDERR_FILE
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=MaxMemPerNode

source ~/.bash_profile

set -e

cd UTS_DIR

srun ./UTS_EXE TREE_ARGS
