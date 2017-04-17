#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 64
#SBATCH -t 00:10:00
#SBATCH -J asyncshmem-isx
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --contiguous

# set -e

ulimit -c unlimited

export PMI_MAX_KVS_ENTRIES=$((1000 * $SLURM_NNODES))
export SMA_OFI_PROVIDER=gni
export LD_LIBRARY_PATH=$SOS_INSTALL/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_ROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/system/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/openshmem/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/sos/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/opt/intel/compilers_and_libraries_2017.1.132/linux/tbb/lib/intel64/gcc4.7/libtbbmalloc.so.2
export HCLIB_LOCALITY_FILE=$HCLIB_HOME/locality_graphs/edison.flat.json
export SOS_INSTALL=$HOME/sandia-shmem-contexts-install
export SOCKETS_PER_NODE=2
export CORES_PER_SOCKET=12

# 2 sockets x 12-core CPUs

# Edison:
export KEYS_PER_CORE=130000000

export SMA_SYMMETRIC_SIZE=$((768 * 1024 * 1024))
srun --ntasks=$(($SLURM_NNODES * 24)) --ntasks-per-node=24 \
    --ntasks-per-socket=12 --cpus-per-task=1 ./bin/isx.weak $KEYS_PER_CORE \
    /tmp/output_weak

export OMP_NUM_THREADS=$CORES_PER_SOCKET
export HCLIB_WORKERS=$CORES_PER_SOCKET
export SMA_SYMMETRIC_SIZE=$((9 * 1024 * 1024 * 1024))

srun --ntasks=$(($SLURM_NNODES * 2)) --ntasks-per-node=2 \
    --ntasks-per-socket=1 --cpus-per-task=$CORES_PER_SOCKET ./bin/isx.omp.weak \
    $(($KEYS_PER_CORE * $CORES_PER_SOCKET)) ./output_weak

srun --ntasks=$(($SLURM_NNODES * 2)) --ntasks-per-node=2 \
    --ntasks-per-socket=1 --cpus-per-task=$CORES_PER_SOCKET ./bin/isx.hclib.weak \
    $(($KEYS_PER_CORE * $CORES_PER_SOCKET)) ./output_weak

srun --ntasks=$(($SLURM_NNODES * 2)) --ntasks-per-node=2 \
    --ntasks-per-socket=1 --cpus-per-task=$CORES_PER_SOCKET ./bin/isx.hclib_contexts.weak \
    $(($KEYS_PER_CORE * $CORES_PER_SOCKET)) ./output_weak

# srun --ntasks=$(($SLURM_NNODES * $PES_PER_NODE)) --ntasks-per-node=$PES_PER_NODE --ntasks-per-socket=1 --cpus-per-task=12 ./bin/isx.hclib.weak $(($KEYS_PER_NODE / $PES_PER_NODE)) /tmp/output_weak
# # srun --no-kill --chdir=/tmp/ --ntasks=$(($SLURM_NNODES * $PES_PER_NODE)) \
# #     --ntasks-per-node=$PES_PER_NODE --ntasks-per-socket=1 --cpus-per-task=12 $(pwd)/bin/isx.hclib_contexts.weak $(($KEYS_PER_NODE / $PES_PER_NODE)) /tmp/output_weak
# 
# echo
# for NODE in $(scontrol show hostname); do
#     echo "Core dumps on $NODE:"
#     srun -N 1 -n 1 --nodelist=$NODE --ntasks-per-node=1 find /tmp/ -name 'core'
# done
# echo
# 
# for NODE in $(scontrol show hostname); do
#     for CORE in $(srun -N 1 -n 1 --nodelist=$NODE --ntasks-per-node=1 find /tmp/ -name 'core'); do
#         LBL=$(basename $(dirname $CORE))
#         srun -N 1 -n 1 --nodelist=$NODE --ntasks-per-node=1 cp $CORE $(pwd)/core.$NODE.$LBL
#     done
# done
# 
