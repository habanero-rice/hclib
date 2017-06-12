#!/bin/bash
#SBATCH --job-name=ISx
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH --contiguous

export LD_PRELOAD=/opt/apps/software/Core/icc/2015.2.164/composer_xe_2015.2.164/tbb/lib/intel64/gcc4.4/libtbbmalloc.so.2

outDir=$SLURM_JOB_ID"-"$SLURM_JOB_NUM_NODES


iterations=2

cd /home/sm108/ISx/ISx-phases
mkdir $outDir


# burn in
export SMA_SYMMETRIC_SIZE=1073741824

oshrun -bind-to socket -N 12 ./bin/isx.weak $((67108864 * 1))  $outDir/out

for p in 1 2 3 4 6 8 12 16 24 32 48 64; do
#for p in 1; do
    export ISX_PHASES=$p
    echo "Flat OpenSHMEM " $ISX_PHASES
    export SMA_SYMMETRIC_SIZE=1073741824
    for (( c=1; c<=$iterations; c++ )); do
        oshrun -bind-to socket -N 12 ./bin/isx.weak $((67108864 * 1)) $outDir/out
    done
done
