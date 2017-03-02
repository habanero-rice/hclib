#!/bin/bash -l

#SBATCH -p production
#SBATCH -N 2
#SBATCH -t 00:05:00
#SBATCH -J hickok-test
#SBATCH --exclusive
#SBATCH --contiguous

set -e

# Each node has 2 sockets, 8 hyperthreaded cores per socket

# for TEST in init shmem_malloc shmem_barrier_all shmem_put64 shmem_broadcast64 shmem_lock_stress shmem_int_wait_until shmem_int_wait_until_any; do
for TEST in init shmem_malloc shmem_barrier_all shmem_put64 shmem_broadcast64; do
    echo ======= $TEST =======
    srun --nodes=2 --ntasks-per-node=2 --ntasks-per-socket=1 ./init
    echo
done
