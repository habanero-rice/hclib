#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 16
#SBATCH -t 00:10:00
#SBATCH -J gups
#SBATCH --exclusive
#SBATCH --contiguous
#SBATCH --mail-type=ALL

# set -e

export PMI_MAX_KVS_ENTRIES=$((1000 * $SLURM_NNODES))
export SMA_SYMMETRIC_SIZE=$((1 * 1024 * 1024 * 1024))
export SMA_OFI_PROVIDER=gni
# export FI_LOG_LEVEL=debug
export SMA_TRAP_ON_ABORT=on

export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib64:$LD_LIBRARY_PATH

ulimit -c unlimited

export NUPDATES=50000000
export TABLE_SIZE_PER_CORE=$((16 * 1024 * 1024))
# export NUPDATES=65536
# export TABLE_SIZE=1000

# srun --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 ./gups -m $TABLE_SIZE -n $NUPDATES -s
# srun --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 ./gups -m $TABLE_SIZE -n $NUPDATES
# echo
# srun --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 ./gups-xor -m $TABLE_SIZE -n $NUPDATES

# srun --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 ./gups -m 100000 -n 100

# 5460 fails, 5465 is fine

# srun --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 ./gups -m $TABLE_SIZE_PER_CORE -s -t 1
# echo
srun --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 ./gups-xor -m $TABLE_SIZE_PER_CORE -s -t 1
echo
srun --ntasks-per-node=2 --ntasks-per-socket=1 --cpus-per-task=12 ./gups-xor-pthreads -m $((12 * $TABLE_SIZE_PER_CORE)) -s -t 12
