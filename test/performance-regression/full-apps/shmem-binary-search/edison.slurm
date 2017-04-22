#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 8
#SBATCH -t 00:10:00
#SBATCH -J binary-search
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --contiguous

set -e

export PMI_MAX_KVS_ENTRIES=$((1000 * $SLURM_NNODES))
export SMA_SYMMETRIC_SIZE=$((1 * 1024 * 1024 * 1024))
export SMA_OFI_PROVIDER=gni
# export FI_LOG_LEVEL=debug

ulimit -c unlimited

export KEYS_PER_CORE=5000

# for EXE in binary-search binary-search-thread-init; do
#     echo "EXE = $EXE"
#     echo
#     for RUN in {1..3}; do
#         srun --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 ./$EXE $KEYS_PER_CORE 1
#     done
#     echo
# done

srun --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 ./binary-search $KEYS_PER_CORE 1
srun --ntasks-per-node=24 --ntasks-per-socket=12 --cpus-per-task=1 ./binary-search-thread-init $KEYS_PER_CORE 1
srun --ntasks-per-node=2 --ntasks-per-socket=1 --cpus-per-task=12 ./binary-search-pthreads $(($KEYS_PER_CORE * 12)) 12
srun --ntasks-per-node=2 --ntasks-per-socket=1 --cpus-per-task=12 ./binary-search-contexts $(($KEYS_PER_CORE * 12)) 12
