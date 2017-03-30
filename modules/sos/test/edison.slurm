#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 2
#SBATCH -t 00:20:00
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

export LD_LIBRARY_PATH=$OFI_HOME/lib:$LD_LIBRARY_PATH

ulimit -c unlimited

export SOS_INSTALL=$HOME/sandia-shmem-contexts-install
cd $HCLIB_HOME/modules/sos/test

# 2 sockets x 12-core CPUs

export PES_PER_NODE=2
export CPUS_PER_PE=1
export HCLIB_WORKERS=12
export NPES=$(($SLURM_NNODES * $PES_PER_NODE))

# for TEST in init many_putmem shmem_malloc shmem_barrier_all \
# 		shmem_broadcast64 shmem_int_async_when shmem_int_async_when_any \
# 		shmem_int_wait_until shmem_int_wait_until_any shmem_lock_stress \
#         shmem_put64; do
for TEST in init many_putmem shmem_malloc shmem_barrier_all shmem_broadcast64; do
    echo "======= Testing $TEST ======="
    srun --ntasks=$NPES --ntasks-per-node=$PES_PER_NODE --cpus-per-task=$CPUS_PER_PE ./$TEST
    echo
done

# # export HCLIB_WORKERS=1
# srun --no-kill --chdir=/tmp/ --ntasks=2 --ntasks-per-node=1 \
#     --cpu_bind=verbose,sockets \
#     --cpus-per-task=$CPUS_PER_PE $HCLIB_HOME/modules/sos/test/shmem_lock_stress
# 
# for NODE in $(scontrol show hostname); do
#     echo "Contents of /tmp/ on $NODE:"
#     srun -N 1 -n 1 --nodelist=$NODE --ntasks-per-node=1 ls -alhrt /tmp/
# done
# 
# for NODE in $(scontrol show hostname); do
#     srun -N 1 -n 1 --nodelist=$NODE --ntasks-per-node=1 cp /tmp/core $(pwd)/core.$NODE
# done
# echo "Done!"
