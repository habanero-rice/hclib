#!/bin/bash

#SBATCH --job-name=GRAPH500-EXE_NAME
#SBATCH --partition=commons
#SBATCH --time=00:10:00
#SBATCH --nodes=NNODES
#SBATCH --exclusive
#SBATCH -o /home/jmg3/degas/hcpp/test/performance-regression/cpu-only/graph500-2.1.4/slurm/output/EXE_NAME.NNODES.stdout
#SBATCH -e /home/jmg3/degas/hcpp/test/performance-regression/cpu-only/graph500-2.1.4/slurm/output/EXE_NAME.NNODES.stderr
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=MaxMemPerNode
#SBATCH --mail-user=jmg3@rice.edu
#SBATCH --mail-type=ALL

# source ~/.bash_profile

set -e

cd /home/jmg3/degas/hcpp/test/performance-regression/cpu-only/graph500-2.1.4
LD_LIBRARY_PATH=/opt/apps/software/Compiler/intel/2015.2.164/impi/5.0.3.048/lib64:$LD_LIBRARY_PATH srun ./mpi/EXE_NAME 15 16
