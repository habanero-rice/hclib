#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 8
#SBATCH -t 00:10:00
#SBATCH -J asyncshmem-g500
#SBATCH --exclusive

# set -e

# Used to be 8 nodes, 10 minutes

ulimit -c unlimited

export PMI_MAX_KVS_ENTRIES=$((1000 * $SLURM_NNODES))
export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_ROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/system/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/openshmem/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/sos/lib:$LD_LIBRARY_PATH

export LD_PRELOAD=/opt/intel/compilers_and_libraries_2017.1.132/linux/tbb/lib/intel64/gcc4.7/libtbbmalloc.so.2
export HCLIB_LOCALITY_FILE=$HCLIB_HOME/locality_graphs/edison.flat.json
export SMA_OFI_PROVIDER=gni
# export FI_LOG_LEVEL=warn

echo "Using OpenSHMEM installation at $OPENSHMEM_INSTALL"

export OMP_NUM_THREADS=1
export HCLIB_WORKERS=$OMP_NUM_THREADS

export WORK_DIR=$(pwd)

# 2 sockets x 12-core CPUs

# Flat MPI
# srun --ntasks=$(($SLURM_NNODES * 16)) --ntasks-per-node=16 --ntasks-per-socket=8 --cpus-per-task=1 $WORK_DIR/../mpi/graph500_mpi_simple $GRAPH_SIZE 16
# srun --ntasks=$(($SLURM_NNODES * 16)) --ntasks-per-node=16 --ntasks-per-socket=8 --cpus-per-task=1 $WORK_DIR/../mpi/graph500_mpi_replicated $GRAPH_SIZE 16

export GRAPH_SIZE=28
export SMA_SYMMETRIC_SIZE=$((6 * 256 * 1024 * 1024)) # Works with GRAPH_SIZE = 29 on bfs_oshmem-single-mailbox-concurrent
# srun --ntasks=$(($SLURM_NNODES * 24)) --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 $WORK_DIR/bfs_oshmem-single-mailbox-concurrent $GRAPH_SIZE 16

# srun --ntasks=$(($SLURM_NNODES * 24)) --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 $WORK_DIR/bfs_oshmem-single-mailbox-concurrent-crc $GRAPH_SIZE 16

# srun --ntasks=$(($SLURM_NNODES * 24)) --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 $WORK_DIR/../mpi/graph500_mpi_simple $GRAPH_SIZE 16

# srun --ntasks=$(($SLURM_NNODES * 12)) --ntasks-per-node=12 --ntasks-per-socket=6 --cpus-per-task=2 $WORK_DIR/../mpi/graph500_mpi_replicated $GRAPH_SIZE 16

srun --ntasks=$(($SLURM_NNODES * 12)) --ntasks-per-node=12 --ntasks-per-socket=6 --cpus-per-task=2 $WORK_DIR/../mpi/graph500_mpi_replicated_csc $GRAPH_SIZE 16

# export GRAPH_SIZE=29
# export SMA_SYMMETRIC_SIZE=$((6 * 256 * 1024 * 1024)) # Works with GRAPH_SIZE = 29 on bfs_oshmem-single-mailbox-concurrent
# srun --ntasks=$(($SLURM_NNODES * 18)) --ntasks-per-node=18 --ntasks-per-socket=9 --cpus-per-task=1 $WORK_DIR/bfs_oshmem-single-mailbox-concurrent-ctx $GRAPH_SIZE 16

# export GRAPH_SIZE=29
# export SMA_SYMMETRIC_SIZE=$((5 * 256 * 1024 * 1024))
# srun --ntasks=$(($SLURM_NNODES * 24)) --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 $WORK_DIR/bfs_oshmem-single-mailbox-concurrent-crc $GRAPH_SIZE 16

# srun --ntasks=$(($SLURM_NNODES * 24)) --ntasks-per-node=24 --cpus-per-task=1 $WORK_DIR/bfs_oshmem-single-mailbox-hiper $GRAPH_SIZE 16
# srun --ntasks=$(($SLURM_NNODES * 24)) --ntasks-per-node=24 --cpus-per-task=1 $WORK_DIR/bfs_oshmem-single-mailbox-concurrent-crc-hiper $GRAPH_SIZE 16
