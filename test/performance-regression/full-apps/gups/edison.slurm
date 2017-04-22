#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 8
#SBATCH -t 00:10:00
#SBATCH -J gups
#SBATCH --exclusive
#SBATCH --contiguous
#SBATCH --mail-type=ALL

set -e

export PMI_MAX_KVS_ENTRIES=$((1000 * $SLURM_NNODES))
export SMA_SYMMETRIC_SIZE=$((1 * 1024 * 1024 * 1024))
export SMA_OFI_PROVIDER=gni
# export FI_LOG_LEVEL=debug

ulimit -c unlimited

# srun --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 ./gups -m $((128 * 1024 * 1024))  -n 1600000000 -s
srun --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 ./gups -m $((128 * 1024 * 1024))  -n 400000000 -s
