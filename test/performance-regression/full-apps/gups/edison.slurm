#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 16
#SBATCH -t 00:20:00
#SBATCH -J gups
#SBATCH --exclusive
#SBATCH --contiguous
#SBATCH --mail-type=ALL
#SBATCH --export=ALL

# set -e

export PMI_MAX_KVS_ENTRIES=$((1000 * $SLURM_NNODES))
export SMA_SYMMETRIC_SIZE=$((1 * 1024 * 1024 * 1024))
export SMA_OFI_PROVIDER=gni
# export FI_LOG_LEVEL=debug
# export SMA_TRAP_ON_ABORT=on

echo "Using OpenSHMEM installation at $OPENSHMEM_INSTALL"

export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib:$LD_LIBRARY_PATH

ulimit -c unlimited

export TABLE_SIZE_PER_CORE=$((32 * 1024 * 1024))

srun --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 ./gups -m $TABLE_SIZE_PER_CORE -s -t 1
echo
srun --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 ./gups-xor -m $TABLE_SIZE_PER_CORE -s -t 1
echo
srun --ntasks-per-node=2 --ntasks-per-socket=1 --cpus-per-task=12 ./gups-xor-pthreads -m $((12 * $TABLE_SIZE_PER_CORE)) -s -t 12
echo
srun --ntasks-per-node=2 --ntasks-per-socket=1 --cpus-per-task=12 ./gups-xor-pthreads-contexts -m $((12 * $TABLE_SIZE_PER_CORE)) -s -t 12
