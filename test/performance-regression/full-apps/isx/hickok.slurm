#!/bin/bash -l

#SBATCH -p connectx5
#SBATCH -N 4
#SBATCH -t 00:10:00
#SBATCH -J asyncshmem-isx
#SBATCH --exclusive

set -e

ulimit -c unlimited

export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib:$LD_LIBRARY_PATH
export LD_PRELOAD="$HOME/gperftools-install/lib/libtcmalloc.so"
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=$((4 * 1024 * 1024 * 1024))

export KEYS_PER_NODE=2000000000

# Each node has 2 sockets, 8 hyperthreaded cores per socket

# Flat OpenSHMEM
export SMA_SYMMETRIC_SIZE=$((1024 * 1024 * 1024))
for RUN in {1..5}; do
    srun --ntasks=$(($SLURM_NNODES * 16)) --ntasks-per-node=16 --ntasks-per-socket=8 --cpus-per-task=1 ./bin/isx.weak $(($KEYS_PER_NODE / 16)) /tmp/output_weak
done

# OpenSHMEM + OMP
export OMP_NUM_THREADS=8
export SMA_SYMMETRIC_SIZE=$((4 * 1024 * 1024 * 1024))
for RUN in {1..5}; do
    srun --ntasks=$(($SLURM_NNODES * 2)) --ntasks-per-node=2 --ntasks-per-socket=1 --cpus-per-task=8 ./bin/isx.omp.weak $(($KEYS_PER_NODE / 2)) /tmp/output_weak
done

# AsyncSHMEM
export HCLIB_WORKERS=$OMP_NUM_THREADS
export SMA_SYMMETRIC_SIZE=$((4 * 1024 * 1024 * 1024))
for RUN in {1..5}; do
    srun --ntasks=$(($SLURM_NNODES * 2)) --ntasks-per-node=2 --ntasks-per-socket=1 --cpus-per-task=8 ./bin/isx.hclib.weak $(($KEYS_PER_NODE / 2)) /tmp/output_weak
done
