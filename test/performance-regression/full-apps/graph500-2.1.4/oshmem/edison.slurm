#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 8
#SBATCH -t 00:20:00
#SBATCH -J asyncshmem-g500
#SBATCH --exclusive
#SBATCH --contiguous
#SBATCH --mail-type=ALL

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
export LD_LIBRARY_PATH=$OFI_HOME/lib:$LD_LIBRARY_PATH

export LD_PRELOAD=/opt/intel/compilers_and_libraries_2017.1.132/linux/tbb/lib/intel64/gcc4.7/libtbbmalloc.so.2
export HCLIB_LOCALITY_FILE=$HCLIB_HOME/locality_graphs/edison.flat.json
export SMA_OFI_PROVIDER=gni
# export FI_LOG_LEVEL=info

echo "Using OpenSHMEM installation at $OPENSHMEM_INSTALL"

export OMP_NUM_THREADS=1
export HCLIB_WORKERS=$OMP_NUM_THREADS

export WORK_DIR=$(pwd)

# 2 sockets x 12-core CPUs

export GRAPH_SIZE=25

### echo
### export OMP_NUM_THREADS=1
### echo Starting MPI version
### export SMA_SYMMETRIC_SIZE=$((6 * 256 * 1024 * 1024))
### time srun --ntasks=$(($SLURM_NNODES * 24)) --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 $WORK_DIR/../mpi/graph500_mpi_simple $GRAPH_SIZE 16
### 
### echo
### export OMP_NUM_THREADS=12
### echo Starting replicated MPI version with $OMP_NUM_THREADS threads per PE
### time srun --ntasks=$(($SLURM_NNODES * 2)) --ntasks-per-node=2 --ntasks-per-socket=1 --cpus-per-task=12 $WORK_DIR/../mpi/graph500_mpi_replicated $GRAPH_SIZE 16
### 
### echo
### export OMP_NUM_THREADS=1
### echo Starting old OpenSHMEM version
### export SMA_SYMMETRIC_SIZE=$((6 * 256 * 1024 * 1024))
### time srun --ntasks=$(($SLURM_NNODES * 24)) --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 $WORK_DIR/bfs_oshmem-single-mailbox-concurrent-crc $GRAPH_SIZE 16

echo
export OMP_NUM_THREADS=12
echo Starting atomics version with $OMP_NUM_THREADS threads per PE
export SMA_SYMMETRIC_SIZE=$((18 * 512 * 1024 * 1024)) # Works with GRAPH_SIZE = 28 on bfs_oshmem-single-mailbox-atomics
time srun --ntasks=$(($SLURM_NNODES * 2)) --ntasks-per-node=2 --ntasks-per-socket=1 --cpus-per-task=12 $WORK_DIR/bfs_oshmem-single-mailbox-atomics $GRAPH_SIZE 16 1

echo
export OMP_NUM_THREADS=12
echo Starting atomics version with $OMP_NUM_THREADS threads per PE
export SMA_SYMMETRIC_SIZE=$((18 * 512 * 1024 * 1024)) # Works with GRAPH_SIZE = 28 on bfs_oshmem-single-mailbox-atomics
time srun --ntasks=$(($SLURM_NNODES * 2)) --ntasks-per-node=2 --ntasks-per-socket=1 --cpus-per-task=12 $WORK_DIR/bfs_oshmem-single-mailbox-atomics-runtime-safety $GRAPH_SIZE 16 1

### echo
### export OMP_NUM_THREADS=12
### echo Starting wavefront version with $OMP_NUM_THREADS threads per PE
### export SMA_SYMMETRIC_SIZE=$((13 * 512 * 1024 * 1024)) # Works with GRAPH_SIZE = 28 on bfs_oshmem-single-mailbox-atomics
### time srun --ntasks=$(($SLURM_NNODES * 2)) --ntasks-per-node=2 --ntasks-per-socket=1 --cpus-per-task=12 $WORK_DIR/bfs_oshmem-single-mailbox-wavefront $GRAPH_SIZE 16 1
