#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 4
#SBATCH -t 00:05:00
#SBATCH -J sos-unit-tests
#SBATCH --exclusive
#SBATCH --mail-type=ALL

# Not sure where this value comes from (or what it means), found experimentally
export PMI_MAX_KVS_ENTRIES=$((1000 * $SLURM_NNODES))
export SMA_SYMMETRIC_SIZE=1073741824
# export SMA_BARRIER_ALGORITHM=linear # tree, dissem, auto
export SMA_SYMMETRIC_HEAP_USE_HUGE_PAGES=true
export SMA_SYMMETRIC_HEAP_PAGE_SIZE=4194304
export SMA_OFI_PROVIDER=gni
export CORES_PER_SOCKET=12
export HCLIB_LOCALITY_FILE=$HOME/hclib/locality_graphs/edison.flat.json
# export HCLIB_INSTRUMENT=1
export LD_PRELOAD=/opt/intel/compilers_and_libraries_2017.1.132/linux/tbb/lib/intel64/gcc4.7/libtbbmalloc.so.2

export LD_LIBRARY_PATH=$OFI_HOME/lib:$LD_LIBRARY_PATH

ulimit -c unlimited

export SOS_INSTALL=$HOME/sandia-shmem-contexts-install
cd $HCLIB_HOME/modules/sos/test

# 2 sockets x 12-core CPUs

export PES_PER_NODE=2
export CPUS_PER_PE=$CORES_PER_SOCKET
export HCLIB_WORKERS=$CPUS_PER_PE
export NPES=$(($SLURM_NNODES * $PES_PER_NODE))

srun --ntasks-per-node=1 rm -r -f '/tmp/hclib.*.dump'

for TEST in init \
        many_putmem \
        shmem_atomics_stress \
        shmem_barrier_all \
        shmem_barrier_stress \
        shmem_broadcast64 \
        shmem_int_async_when \
        shmem_int_wait_until \
        shmem_lock_stress \
        shmem_malloc \
        shmem_put64; do
    echo "======= Testing $TEST ======="
    srun --no-kill --chdir=/tmp/ --ntasks=$NPES --ntasks-per-node=$PES_PER_NODE \
        --cpus-per-task=$CPUS_PER_PE $(pwd)/$TEST
    echo
done

# for NODE in $(scontrol show hostname); do
#     echo "HClib dumps and core dumps on $NODE:"
#     srun -N 1 -n 1 --nodelist=$NODE --ntasks-per-node=1 find /tmp/ -name 'hclib.*.dump'
#     srun -N 1 -n 1 --nodelist=$NODE --ntasks-per-node=1 find /tmp/ -name '*core*'
# done
# echo
# 
# for NODE in $(scontrol show hostname); do
#     for CORE in $(srun -N 1 -n 1 --nodelist=$NODE --ntasks-per-node=1 find /tmp/ -name 'hclib.*.dump') \
#             $(srun -N 1 -n 1 --nodelist=$NODE --ntasks-per-node=1 find /tmp/ -name '*core*'); do
#         srun -N 1 -n 1 --nodelist=$NODE --ntasks-per-node=1 cp -r $CORE $(pwd)/$NODE.$(basename $CORE)
#     done
# done
