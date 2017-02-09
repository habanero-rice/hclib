#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:05:00
#SBATCH -J hclib-micro-apps
#SBATCH --exclusive
#SBATCH --mail-type=ALL

ulimit -c unlimited
export HCLIB_WORKERS=24

# 2 sockets x 12-core CPUs

cd $HOME/hclib/test/performance-regression/micro

for TEST in task_spawn future_spawn task_wait_flat task_wait_recursive fan_out; do
    echo "========== Testing ${TEST} =========="
    echo

    for EXE in ${TEST}_hclib ${TEST}_omp; do
        echo "========== $EXE =========="
        srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=24 \
            --cpu_bind=verbose,sockets ./$EXE
    done
done
