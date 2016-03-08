#!/bin/bash

set -e

CURR_DIR=$(pwd)

if [[ -z "$HCLIB_CPU_BENCHMARK_REF_DIR" ]]; then
    echo Please set HCLIB_CPU_BENCHMARK_REF_DIR to be the directory to store \
        the reference benchmarks at
    echo Downloading and compiling the reference benchmarks can consume \
        significant disk space, so we allow users to set a custom installation \
        directory
    exit 1
fi

mkdir -p $HCLIB_CPU_BENCHMARK_REF_DIR

if [[ ! -f $HCLIB_CPU_BENCHMARK_REF_DIR/rodinia_3.1.tar.bz2 ]]; then
    wget --directory-prefix=$HCLIB_CPU_BENCHMARK_REF_DIR http://www.cs.virginia.edu/~kw5na/lava/Rodinia/Packages/Current/rodinia_3.1.tar.bz2
fi

if [[ ! -d $HCLIB_CPU_BENCHMARK_REF_DIR/rodinia_3.1 ]]; then
    cd $HCLIB_CPU_BENCHMARK_REF_DIR && tar xf rodinia_3.1.tar.bz2 && cd $CURR_DIR
fi

# Don't build the mummergpu OpenMP example, Makefile seems broken
sed -i.bak '/openmp\/mummergpu/d' $HCLIB_CPU_BENCHMARK_REF_DIR/rodinia_3.1/Makefile
cd $HCLIB_CPU_BENCHMARK_REF_DIR/rodinia_3.1 && make clean && make OMP && cd $CURR_DIR

N_EXES=$(ls -l $HCLIB_CPU_BENCHMARK_REF_DIR/rodinia_3.1/bin/linux/omp/ | wc -l)
if [[ $N_EXES != 20 ]]; then
    echo Unexpected number of executables created, $N_EXES. Expected 20.
    ls -l $HCLIB_CPU_BENCHMARK_REF_DIR/rodinia_3.1/bin/linux/omp/
    exit 1
fi
