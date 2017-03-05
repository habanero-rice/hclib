#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:04:00
#SBATCH -J hclib-c-tests
#SBATCH --exclusive
#SBATCH --mail-type=ALL

ulimit -c unlimited

cd $HCLIB_HOME/test/c

# 2 sockets x 12-core CPUs
# srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=24 \
#     --cpu_bind=verbose,sockets ./test_all.sh
export HCLIB_WORKERS=24
srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=24 --cpu_bind=verbose,mask_cpu:FFFFFF ./async0
ERR=$?

echo
echo Finished with exit code $ERR
