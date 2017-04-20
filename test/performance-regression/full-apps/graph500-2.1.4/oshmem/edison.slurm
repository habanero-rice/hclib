#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 8
#SBATCH -t 00:30:00
#SBATCH -J asyncshmem-g500
#SBATCH --exclusive

# set -e

# Used to be 8 nodes, 10 minutes

ulimit -c unlimited

export PMI_MAX_KVS_ENTRIES=$((1000 * $SLURM_NNODES))
export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_ROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/system/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/openshmem/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/sos/lib:$LD_LIBRARY_PATH

export LD_PRELOAD=/opt/intel/compilers_and_libraries_2017.1.132/linux/tbb/lib/intel64/gcc4.7/libtbbmalloc.so.2
export HCLIB_LOCALITY_FILE=$HCLIB_HOME/locality_graphs/edison.flat.json
export SMA_OFI_PROVIDER=gni
# export FI_LOG_LEVEL=warn

export OMP_NUM_THREADS=1
export HCLIB_WORKERS=$OMP_NUM_THREADS

export WORK_DIR=$(pwd)

# 2 sockets x 12-core CPUs

# Flat MPI
# srun --ntasks=$(($SLURM_NNODES * 16)) --ntasks-per-node=16 --ntasks-per-socket=8 --cpus-per-task=1 $WORK_DIR/../mpi/graph500_mpi_simple $GRAPH_SIZE 16
# srun --ntasks=$(($SLURM_NNODES * 16)) --ntasks-per-node=16 --ntasks-per-socket=8 --cpus-per-task=1 $WORK_DIR/../mpi/graph500_mpi_replicated $GRAPH_SIZE 16

export GRAPH_SIZE=29
# export SMA_SYMMETRIC_SIZE=$((5 * 256 * 1024 * 1024)) # Works with GRAPH_SIZE = 29 on bfs_oshmem-single-mailbox-concurrent
# srun --ntasks=$(($SLURM_NNODES * 24)) --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 $WORK_DIR/bfs_oshmem-single-mailbox-concurrent $GRAPH_SIZE 16
export SMA_SYMMETRIC_SIZE=$((6 * 256 * 1024 * 1024)) # Works with GRAPH_SIZE = 29 on bfs_oshmem-single-mailbox-concurrent
srun --ntasks=$(($SLURM_NNODES * 18)) --ntasks-per-node=18 --ntasks-per-socket=9 --cpus-per-task=1 $WORK_DIR/bfs_oshmem-single-mailbox-concurrent $GRAPH_SIZE 16

export GRAPH_SIZE=29
export SMA_SYMMETRIC_SIZE=$((6 * 256 * 1024 * 1024)) # Works with GRAPH_SIZE = 29 on bfs_oshmem-single-mailbox-concurrent
srun --ntasks=$(($SLURM_NNODES * 18)) --ntasks-per-node=18 --ntasks-per-socket=9 --cpus-per-task=1 $WORK_DIR/bfs_oshmem-single-mailbox-concurrent-ctx $GRAPH_SIZE 16

# export GRAPH_SIZE=29
# export SMA_SYMMETRIC_SIZE=$((5 * 256 * 1024 * 1024))
# srun --ntasks=$(($SLURM_NNODES * 24)) --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 $WORK_DIR/bfs_oshmem-single-mailbox-concurrent-crc $GRAPH_SIZE 16

# srun --ntasks=$(($SLURM_NNODES * 24)) --ntasks-per-node=24 --cpus-per-task=1 $WORK_DIR/bfs_oshmem-single-mailbox-hiper $GRAPH_SIZE 16
# srun --ntasks=$(($SLURM_NNODES * 24)) --ntasks-per-node=24 --cpus-per-task=1 $WORK_DIR/bfs_oshmem-single-mailbox-concurrent-crc-hiper $GRAPH_SIZE 16

# for NODE in $(scontrol show hostname); do
#     echo "Core dumps on $NODE:"
#     srun -N 1 -n 1 --nodelist=$NODE --ntasks-per-node=1 find /tmp/ -name 'core'
# done
# for NODE in $(scontrol show hostname); do
#     for CORE in $(srun -N 1 -n 1 --nodelist=$NODE --ntasks-per-node=1 find /tmp/ -name 'core'); do
#         LBL=$(basename $(dirname $CORE))
#         srun -N 1 -n 1 --nodelist=$NODE --ntasks-per-node=1 cp $CORE $(pwd)/core.$NODE.$LBL
#     done
# done
# echo
