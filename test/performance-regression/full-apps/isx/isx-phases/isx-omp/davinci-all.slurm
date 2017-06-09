#!/bin/bash
#SBATCH --job-name=ISx
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH --contiguous

export LD_PRELOAD=/opt/apps/software/Core/icc/2015.2.164/composer_xe_2015.2.164/tbb/lib/intel64/gcc4.4/libtbbmalloc.so.2

outDir=$SLURM_JOB_ID"-"$SLURM_JOB_NUM_NODES


iterations=2

cd /home/sm108/COMP590
mkdir $outDir

# burn in
oshrun -bind-to socket -N 2 ./bin/isx.weak $((67108864 * 6))

for p in 1 2 3 4 6 8 12 16 24 32 48 64; do
    export ISX_PHASES=$p
    echo "OMP - new" $ISX_PHASES
    for (( c=1; c<=$iterations; c++ )); do
        export SMA_SYMMETRIC_SIZE=4294967296
        export OMP_NUM_THREADS=6
        export HCLIB_WORKERS=6
        oshrun -bind-to socket -N 2 ./bin/isx.weak $((67108864 * 6)) $outDir/out
    done
done


cd $HOME/libs/hclib/test/performance-regression/full-apps/isx
mkdir $outDir

echo "Flat OpenSHMEM"
for (( i=1; i<=$iterations; i++ )); do
    export SMA_SYMMETRIC_SIZE=1073741824
    oshrun -bind-to core -N 12 ./bin/isx.weak 67108864 $outDir/out
done

echo 
echo "AsyncSHMEM"
for (( i=1; i<=$iterations; i++ )); do
    export SMA_SYMMETRIC_SIZE=4294967296
    export OMP_NUM_THREADS=6
    export HCLIB_WORKERS=6
    oshrun -bind-to socket -N 2 ./bin_hclib/isx.weak $((67108864 * 6)) $outDir/out-hclib-2
done

echo
echo "OpenMP + OpenSHMEM"
for (( i=1; i<=$iterations; i++ )); do
    export SMA_SYMMETRIC_SIZE=4294967296
    export OMP_NUM_THREADS=6
    export HCLIB_WORKERS=6
    oshrun -bind-to socket -N 2 ./bin_omp/isx.weak $((67108864 * 6)) $outDir/out-omp2-2
done
