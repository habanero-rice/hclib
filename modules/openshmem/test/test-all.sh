#!/bin/bash

#SBATCH --job-name=HCLIB-PERFORMANCE-REGRESSION
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000m
#SBATCH --time=00:05:00
#SBATCH --mail-user=jmg3@rice.edu
#SBATCH --mail-type=ALL
#SBATCH --export=ALL
#SBATCH --partition=commons
#SBATCH --exclusive

if [[ -z "$HCLIB_LOCALITY_FILE" ]]; then
    echo HCLIB_LOCALITY_FILE must be set for the OpenSHMEM tests
    exit 1
fi

make clean
make
rm -f samplesort_omp

for FILE in $(ls); do
    if [[ -x $FILE && $FILE != "test-all.sh" ]]; then
        echo "======= Running $FILE ======="
        srun --exclusive --tasks-per-node=1 --mem=MaxMemPerNode -N 4 -p commons -t 00:02:00 ./$FILE
    fi
done
