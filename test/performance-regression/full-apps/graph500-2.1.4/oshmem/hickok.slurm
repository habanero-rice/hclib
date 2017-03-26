#!/bin/bash -l

#SBATCH -p connectx5
#SBATCH -N 32
#SBATCH -t 00:20:00
#SBATCH -J asyncshmem-g500
#SBATCH --exclusive

set -e

ulimit -c unlimited

export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib:$LD_LIBRARY_PATH
export LD_PRELOAD="$HOME/gperftools-install/lib/libtcmalloc.so"
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=$((4 * 1024 * 1024 * 1024))

export GRAPH_SIZE=27

export OMP_NUM_THREADS=1
export HCLIB_WORKERS=$OMP_NUM_THREADS

# Each node has 2 sockets, 8 hyperthreaded cores per socket

srun --ntasks-per-node=1 mkdir -p /var/tmp/$USER

# Flat OpenSHMEM
export SMA_SYMMETRIC_SIZE=$((2 * 1024 * 1024 * 1024))
# for RUN in {1..5}; do
    srun --chdir=/var/tmp/$USER --ntasks=$(($SLURM_NNODES * 16)) --ntasks-per-node=16 --ntasks-per-socket=8 --cpus-per-task=1 ../mpi/graph500_mpi_simple $GRAPH_SIZE 16
# done
# for RUN in {1..5}; do
    srun --chdir=/var/tmp/$USER --ntasks=$(($SLURM_NNODES * 16)) --ntasks-per-node=16 --ntasks-per-socket=8 --cpus-per-task=1 ../mpi/graph500_mpi_replicated $GRAPH_SIZE 16
# done
# for RUN in {1..5}; do
    srun --chdir=/var/tmp/$USER --ntasks=$(($SLURM_NNODES * 16)) --ntasks-per-node=16 --ntasks-per-socket=8 --cpus-per-task=1 ./bfs_oshmem-single-mailbox-concurrent $GRAPH_SIZE 16
# done
# for RUN in {1..5}; do
    srun --chdir=/var/tmp/$USER --ntasks=$(($SLURM_NNODES * 16)) --ntasks-per-node=16 --ntasks-per-socket=8 --cpus-per-task=1 ./bfs_oshmem-single-mailbox-concurrent-crc $GRAPH_SIZE 16
# done
# for RUN in {1..5}; do
    srun --chdir=/var/tmp/$USER --ntasks=$(($SLURM_NNODES * 12)) --ntasks-per-node=12 --ntasks-per-socket=6 --cpus-per-task=1 ./bfs_oshmem-single-mailbox-hiper $GRAPH_SIZE 16
# done
# for RUN in {1..5}; do
    srun --chdir=/var/tmp/$USER --ntasks=$(($SLURM_NNODES * 12)) --ntasks-per-node=12 --ntasks-per-socket=6 --cpus-per-task=1 ./bfs_oshmem-single-mailbox-concurrent-crc-hiper $GRAPH_SIZE 16
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
# for RUN in {1..5}; do
#     srun --ntasks=$(($SLURM_NNODES * 2)) --ntasks-per-node=2 --ntasks-per-socket=1 --cpus-per-task=8 ./uts-hclib-shmem-naive $T1XXL
# done
