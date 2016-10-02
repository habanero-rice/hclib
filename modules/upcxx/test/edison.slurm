#!/bin/bash -l
#SBATCH -p debug
#SBATCH -N 4
#SBATCH -t 2:00
#SBATCH -J hpgmg
#SBATCH --exclusive
#SBATCH --exclusive
#SBATCH --mail-type=ALL

export GASNET_MAX_SEGSIZE='4096MB'
export GASNET_PHYSMEM_MAX=16G
export GASNET_BACKTRACE=1
export HCLIB_LOCALITY_FILE=$HCLIB_HOME/locality_graphs/edison.json
# export HCLIB_LOCALITY_FILE=$HCLIB_HOME/locality_graphs/edison.no_interconnect.json
export OMP_NUM_THREADS=12
export HCLIB_WORKERS=$OMP_NUM_THREADS

cd $HCLIB_HOME/modules/upcxx/test
LD_LIBRARY_PATH=$HCLIB_HOME/modules/mpi/lib:$LD_LIBRARY_PATH srun --ntasks=8 \
    --ntasks-per-node=2 --ntasks-per-socket=1 --cpus-per-task=12 \
    --cpu_bind=verbose,cores ./active_msg
