#!/bin/bash -l

#SBATCH -p regular
#SBATCH -N 16
#SBATCH -t 00:30:00
#SBATCH -J asyncshmem-g500
#SBATCH --exclusive
#SBATCH --contiguous
#SBATCH --mail-type=ALL

# set -e

ulimit -c unlimited

export PMI_MAX_KVS_ENTRIES=$((1000 * $SLURM_NNODES))
export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_ROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/system/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/openshmem/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/sos/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OFI_HOME/lib:$LD_LIBRARY_PATH

export LD_PRELOAD=/opt/intel/compilers_and_libraries_2017.1.132/linux/tbb/lib/intel64/gcc4.7/libtbbmalloc.so.2
export HCLIB_LOCALITY_FILE=$HCLIB_HOME/locality_graphs/edison.flat.json
export SMA_OFI_PROVIDER=gni
# export FI_LOG_LEVEL=info

export HCLIB_WORKERS=$OMP_NUM_THREADS

export WORK_DIR=$(pwd)

# 2 sockets x 16 cores per socket for Cori Haswell
# 2 sockets x 12 cores per socket for Edison
export CORES_PER_SOCKET=12
export SOCKETS_PER_NODE=2
export CORES_PER_NODE=$(($SOCKETS_PER_NODE * $CORES_PER_SOCKET))

# export GRAPH_SIZE=30
export GRAPH_SIZE=29

# export OMP_NUM_THREADS=1
# time srun --ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) \
#     --ntasks-per-node=$CORES_PER_NODE --ntasks-per-socket=$CORES_PER_SOCKET \
#     --cpus-per-task=1 $WORK_DIR/../mpi/graph500_mpi_simple $GRAPH_SIZE 16
# echo

# export OMP_NUM_THREADS=$CORES_PER_SOCKET
# time srun --ntasks=$(($SLURM_NNODES * $SOCKETS_PER_NODE)) \
#     --ntasks-per-node=$SOCKETS_PER_NODE --ntasks-per-socket=1 \
#     --cpus-per-task=$CORES_PER_SOCKET $WORK_DIR/../mpi/graph500_mpi_replicated $GRAPH_SIZE 16
# echo

# export OMP_NUM_THREADS=$CORES_PER_SOCKET
# time srun --ntasks=$(($SLURM_NNODES * $SOCKETS_PER_NODE)) \
#     --ntasks-per-node=$SOCKETS_PER_NODE --ntasks-per-socket=1 \
#     --cpus-per-task=$CORES_PER_SOCKET $WORK_DIR/../mpi/graph500_mpi_replicated_csc $GRAPH_SIZE 16
# echo

# export OMP_NUM_THREADS=1
# export SMA_SYMMETRIC_SIZE=$((12 * 128 * 1024 * 1024))
# time srun --ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) \
#     --ntasks-per-node=$CORES_PER_NODE --ntasks-per-socket=$CORES_PER_SOCKET \
#     --cpus-per-task=1 $WORK_DIR/../oshmem/bfs_oshmem-single-mailbox-concurrent $GRAPH_SIZE 16
# echo

# export OMP_NUM_THREADS=1
# export SMA_SYMMETRIC_SIZE=$((12 * 128 * 1024 * 1024))
# time srun --ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) \
#     --ntasks-per-node=$CORES_PER_NODE --ntasks-per-socket=$CORES_PER_SOCKET \
#     --cpus-per-task=1 $WORK_DIR/../oshmem/bfs_oshmem-single-mailbox-concurrent-crc $GRAPH_SIZE 16
# echo
# 
# export OMP_NUM_THREADS=$CORES_PER_SOCKET
# export SMA_SYMMETRIC_SIZE=$((80 * 128 * 1024 * 1024))
# time srun --ntasks=$(($SLURM_NNODES * $SOCKETS_PER_NODE)) \
#     --ntasks-per-node=$SOCKETS_PER_NODE --ntasks-per-socket=1 \
#     --cpus-per-task=$CORES_PER_SOCKET $WORK_DIR/../oshmem/bfs_oshmem-single-mailbox-wavefront $GRAPH_SIZE 16 1
# echo
# 
# export OMP_NUM_THREADS=$CORES_PER_SOCKET
# export SMA_SYMMETRIC_SIZE=$((80 * 128 * 1024 * 1024))
# time srun --ntasks=$(($SLURM_NNODES * $SOCKETS_PER_NODE)) \
#     --ntasks-per-node=$SOCKETS_PER_NODE --ntasks-per-socket=1 \
#     --cpus-per-task=$CORES_PER_SOCKET $WORK_DIR/../oshmem/bfs_oshmem-single-mailbox-atomics $GRAPH_SIZE 16 1
# echo

export OMP_NUM_THREADS=2
export SMA_SYMMETRIC_SIZE=$((12 * 128 * 1024 * 1024))
time srun --ntasks=$(($SLURM_NNODES * $CORES_PER_NODE / 2)) \
    --ntasks-per-node=$(($CORES_PER_NODE / 2)) --ntasks-per-socket=$(($CORES_PER_SOCKET / 2)) \
    --cpus-per-task=2 $WORK_DIR/../oshmem/bfs_oshmem-single-mailbox-atomics $GRAPH_SIZE 16 1
echo
