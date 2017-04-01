#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 32
#SBATCH -t 00:05:00
#SBATCH -J asyncshmem-isx
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --contiguous

set -e

ulimit -c unlimited

export PMI_MAX_KVS_ENTRIES=$((1000 * $SLURM_NNODES))
export SMA_OFI_PROVIDER=gni
export LD_LIBRARY_PATH=$SOS_INSTALL/lib:$LD_LIBRARY_PATH
export KEYS_PER_NODE=1000000000
export LD_PRELOAD=/opt/intel/compilers_and_libraries_2017.1.132/linux/tbb/lib/intel64/gcc4.7/libtbbmalloc.so.2
export HCLIB_LOCALITY_FILE=$HCLIB_HOME/locality_graphs/edison.flat.json

export SOS_INSTALL=$HOME/sandia-shmem-contexts-install

# 2 sockets x 12-core CPUs

export PES_PER_NODE=2
export NPES=$(($PES_PER_NODE * $SLURM_NNODES))
export CPUS_PER_PE=12
export OMP_NUM_THREADS=$CPUS_PER_PE
export HCLIB_WORKERS=$OMP_NUM_THREADS
export SMA_SYMMETRIC_SIZE=$((4 * 1024 * 1024 * 1024))

srun --ntasks=$(($SLURM_NNODES * $PES_PER_NODE)) --ntasks-per-node=$PES_PER_NODE --ntasks-per-socket=1 --cpus-per-task=12 ./bin/isx.hclib.weak $(($KEYS_PER_NODE / $PES_PER_NODE)) /tmp/output_weak
srun --ntasks=$(($SLURM_NNODES * $PES_PER_NODE)) --ntasks-per-node=$PES_PER_NODE --ntasks-per-socket=1 --cpus-per-task=12 ./bin/isx.hclib_contexts.weak $(($KEYS_PER_NODE / $PES_PER_NODE)) /tmp/output_weak
