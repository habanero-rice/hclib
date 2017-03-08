#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 2
#SBATCH -t 00:02:00
#SBATCH -J sos-unit-tests
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --contiguous

# Not sure where this value comes from (or what it means), found experimentally
export PMI_MAX_KVS_ENTRIES=$((1000 * $SLURM_NNODES))
export SMA_SYMMETRIC_SIZE=1073741824
# export SMA_BARRIER_ALGORITHM=linear # tree, dissem, auto
export SMA_SYMMETRIC_HEAP_USE_HUGE_PAGES=true
export SMA_SYMMETRIC_HEAP_PAGE_SIZE=4194304
export SMA_OFI_PROVIDER=gni
export CORES_PER_SOCKET=12

ulimit -c unlimited

export SOS_INSTALL=$HOME/sandia-shmem-contexts-install
cd $HCLIB_HOME/modules/sos/test

# 2 sockets x 12-core CPUs

export PES_PER_NODE=2
export CPUS_PER_PE=12
export HCLIB_WORKERS=$CPUS_PER_PE
export NPES=$(($SLURM_NNODES * $PES_PER_NODE))

for TEST in init many_putmem; do
    echo "======= Testing $TEST ======="
    srun --ntasks=$NPES --ntasks-per-node=$PES_PER_NODE --cpus-per-task=$CPUS_PER_PE ./$TEST
    echo
done
