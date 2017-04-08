#!/bin/bash -l

#SBATCH -p commons
#SBATCH -N 4
#SBATCH -t 00:10:00
#SBATCH -J asyncshmem-isx
#SBATCH --exclusive
#SBATCH --contiguous

set -e

ulimit -c unlimited

export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/opt/apps/software/Core/icc/2015.2.164/composer_xe_2015.2.164/tbb/lib/intel64/gcc4.4/libtbbmalloc.so.2

export KEYS_PER_CORE=67108864

# Each node has 2 sockets, 8 hyperthreaded cores per socket

# Flat OpenSHMEM
export SMA_SYMMETRIC_SIZE=$((1 * 1024 * 1024 * 1024))
oshrun -bind-to core -N 12 ./bin/isx.weak $KEYS_PER_CORE ./output_weak.flat

# OpenSHMEM + OMP
export OMP_NUM_THREADS=6
export SMA_SYMMETRIC_SIZE=$((8 * 1024 * 1024 * 1024))
oshrun -bind-to socket -N 2 ./bin/isx.omp.weak $(($KEYS_PER_CORE * 6)) ./output_weak.omp

# AsyncSHMEM
# export HCLIB_WORKERS=6
# export SMA_SYMMETRIC_SIZE=2684354560
# for RUN in {1..5}; do
#     oshrun -bind-to socket -N 2 ./bin/isx.hclib.weak $(($KEYS_PER_NODE / 2)) /tmp/output_weak
# done
