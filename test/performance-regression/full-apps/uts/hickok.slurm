#!/bin/bash -l

#SBATCH -p connectx5
#SBATCH -N 8
#SBATCH -t 00:10:00
#SBATCH -J asyncshmem-uts
#SBATCH --exclusive

set -e

ulimit -c unlimited

export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib:$LD_LIBRARY_PATH
export LD_PRELOAD="$HOME/gperftools-install/lib/libtcmalloc.so"
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=$((4 * 1024 * 1024 * 1024))

source sample_trees.sh

# Each node has 2 sockets, 8 hyperthreaded cores per socket

# Flat OpenSHMEM
export SMA_SYMMETRIC_SIZE=$((1024 * 1024 * 1024))
# for RUN in {1..5}; do
#     srun --ntasks=$(($SLURM_NNODES * 16)) --ntasks-per-node=16 --ntasks-per-socket=8 --cpus-per-task=1 ./uts-shmem $T1XXL
# done

# OpenSHMEM + OMP
export OMP_NUM_THREADS=8
export SMA_SYMMETRIC_SIZE=$((4 * 1024 * 1024 * 1024))
# for RUN in {1..5}; do
#     srun --ntasks=$(($SLURM_NNODES * 2)) --ntasks-per-node=2 --ntasks-per-socket=1 --cpus-per-task=8 ./uts-shmem-omp $T1XXL
# done

# AsyncSHMEM
export HCLIB_WORKERS=$OMP_NUM_THREADS
export SMA_SYMMETRIC_SIZE=$((4 * 1024 * 1024 * 1024))
for RUN in {1..5}; do
    srun --ntasks=$(($SLURM_NNODES * 2)) --ntasks-per-node=2 --ntasks-per-socket=1 --cpus-per-task=8 ./uts-hclib-shmem-naive $T1XXL
done
