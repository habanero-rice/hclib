#!/bin/bash -l

#SBATCH -p commons
#SBATCH -N 32
#SBATCH -t 00:20:00
#SBATCH -J asyncshmem-g500
#SBATCH --exclusive

set -e

ulimit -c unlimited

export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/opt/apps/software/Core/icc/2015.2.164/composer_xe_2015.2.164/tbb/lib/intel64/gcc4.4/libtbbmalloc.so.2

export GRAPH_SIZE=25

export OMP_NUM_THREADS=1
export HCLIB_WORKERS=$OMP_NUM_THREADS

# Each node has 2 sockets, 8 hyperthreaded cores per socket

# Flat OpenSHMEM
export SMA_SYMMETRIC_SIZE=$((1024 * 1024 * 1024))
$OPENSHMEM_INSTALL/bin/mpirun -x UCX_NET_DEVICES=mlx4_0:1 --map-by core ../mpi/graph500_mpi_simple $GRAPH_SIZE 16
# $OPENSHMEM_INSTALL/bin/mpirun -x UCX_NET_DEVICES=mlx4_0:1 --map-by core ../mpi/graph500_mpi_replicated $GRAPH_SIZE 16
$OPENSHMEM_INSTALL/bin/oshrun -x UCX_NET_DEVICES=mlx4_0:1 --map-by core ./bfs_oshmem-single-mailbox-concurrent $GRAPH_SIZE 16
$OPENSHMEM_INSTALL/bin/oshrun -x UCX_NET_DEVICES=mlx4_0:1 --map-by core ./bfs_oshmem-single-mailbox-concurrent-crc $GRAPH_SIZE 16
$OPENSHMEM_INSTALL/bin/oshrun -x UCX_NET_DEVICES=mlx4_0:1 --map-by core ./bfs_oshmem-single-mailbox-hiper $GRAPH_SIZE 16
$OPENSHMEM_INSTALL/bin/oshrun -x UCX_NET_DEVICES=mlx4_0:1 --map-by core ./bfs_oshmem-single-mailbox-concurrent-crc-hiper $GRAPH_SIZE 16

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
