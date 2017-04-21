#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 16
#SBATCH -t 00:05:00
#SBATCH -J mandelbrot
#SBATCH --exclusive
#SBATCH --mail-type=ALL

export PMI_MAX_KVS_ENTRIES=$((1000 * $SLURM_NNODES))
export SMA_SYMMETRIC_SIZE=$((5 * 1024 * 1024 * 1024))
export SMA_OFI_PROVIDER=gni
# export FI_LOG_LEVEL=debug

ulimit -c unlimited

for THREADS in 12; do
    for BLOCKING in on off; do
        for CONTEXTS in no_contexts contexts pipelining; do
            for BALANCE_GRANULARITY in 256 512 768 1024 2048; do

                ARGS="-t $THREADS -w 32768 -h 32768 -j $BALANCE_GRANULARITY"

                if [[ $CONTEXTS == contexts ]]; then
                    ARGS="$ARGS -c"
                elif [[ $CONTEXTS == pipelining ]]; then
                    ARGS="$ARGS -p"
                fi

                if [[ $BLOCKING == on ]]; then
                    ARGS="$ARGS -b"
                fi

                echo THREADS = $THREADS , BLOCKING = $BLOCKING , CONTEXTS = $CONTEXTS , BALANCE_GRANULARITY = $BALANCE_GRANULARITY
                srun --ntasks-per-node=1 --cpus-per-task=24 ./mandelbrot $ARGS
                echo
            done
        done
    done
done
