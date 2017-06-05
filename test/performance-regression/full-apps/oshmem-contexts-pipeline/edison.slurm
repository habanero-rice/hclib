#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 4
#SBATCH -t 00:02:00
#SBATCH -J shmem_ctx_pipelined_reduce
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --contiguous

export PMI_MAX_KVS_ENTRIES=$((1000 * $SLURM_NNODES))
export SMA_SYMMETRIC_SIZE=$((1024 * 1024 * 1024))
export SMA_OFI_PROVIDER=gni
# export FI_LOG_PROV=gni
# export FI_LOG_LEVEL=info
# export FI_LOG_SUBSYS=ep_data

ulimit -c unlimited

# srun --ntasks-per-node=24 --ntasks-per-socket=12 ./shmem_ctx_pipelined_reduce

for EXE in shmem_ctx_pipelined_reduce shmem_ctx_pipelined_reduce_no_context; do
    echo Running $EXE
    srun --ntasks-per-node=12 --ntasks-per-socket=6 ./$EXE
    echo
done
