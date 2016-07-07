#!/bin/bash

declare -A DATASETS
# DATASETS["rodinia/backprop/backprop,small"]="65536"
# DATASETS["rodinia/backprop/backprop,large"]="4194304"
# DATASETS["rodinia/bfs/bfs,small"]="4 $RODINIA_DATA_DIR/bfs/graph1MW_6.txt"
# DATASETS["rodinia/bfs/bfs,large"]="4 $RODINIA_DATA_DIR/bfs/graph16M.txt"
# DATASETS["rodinia/b+tree/b+tree.out,large"]="core 2 file $RODINIA_DATA_DIR/b+tree/mil.txt command $RODINIA_DATA_DIR/b+tree/command.txt"
# DATASETS["rodinia/cfd/euler3d_cpu_double,large"]="rodinia/cfd/fvcorr.domn.193K"
# DATASETS["rodinia/heartwall/heartwall,large"]="$RODINIA_DATA_DIR/heartwall/test.avi 20 4"
# DATASETS["rodinia/hotspot/hotspot,small"]="1024 1024 2 4 $RODINIA_DATA_DIR/hotspot/temp_1024 $RODINIA_DATA_DIR/hotspot/power_1024 $RODINIA_DATA_DIR/hotspot/output.out"
# DATASETS["rodinia/hotspot/hotspot,large"]="8192 8192 2 4 $RODINIA_DATA_DIR/hotspot/temp_8192 $RODINIA_DATA_DIR/hotspot/power_8192 $RODINIA_DATA_DIR/hotspot/output.out"
# DATASETS["rodinia/hotspot3D/3D,small"]="64 8 100 $RODINIA_DATA_DIR/hotspot3D/power_64x8 $RODINIA_DATA_DIR/hotspot3D/temp_64x8 output.out"
# DATASETS["rodinia/hotspot3D/3D,large"]="512 8 100 $RODINIA_DATA_DIR/hotspot3D/power_512x8 $RODINIA_DATA_DIR/hotspot3D/temp_512x8 output.out"
# DATASETS["rodinia/kmeans/kmeans,large"]="-i $RODINIA_DATA_DIR/kmeans/kdd_cup"
# DATASETS["rodinia/lavaMD/lavaMD,small"]="-cores 4 -boxes1d 10"
# DATASETS["rodinia/lavaMD/lavaMD,large"]="-cores 4 -boxes1d 20"
# DATASETS["rodinia/leukocyte/OpenMP/leukocyte,small"]="5 4 $RODINIA_DATA_DIR/leukocyte/testfile.avi"
# DATASETS["rodinia/leukocyte/OpenMP/leukocyte,large"]="20 4 $RODINIA_DATA_DIR/leukocyte/testfile.avi"
# DATASETS["rodinia/lud/omp/lud_omp,small"]="-s 4000"
# DATASETS["rodinia/lud/omp/lud_omp,large"]="-s 8000"
# DATASETS["rodinia/nw/needle,small"]="2048 10 2"
# DATASETS["rodinia/nw/needle,large"]="32768 10 2"
# DATASETS["rodinia/particlefilter/particle_filter,small"]="-x 128 -y 128 -z 128 -np 10000"
# DATASETS["rodinia/particlefilter/particle_filter,large"]="-x 256 -y 256 -z 256 -np 20000"
# DATASETS["rodinia/pathfinder/pathfinder,small"]="10000 100"
# DATASETS["rodinia/pathfinder/pathfinder,large"]="10000000 100"
# DATASETS["rodinia/srad/srad,small"]="2048 2048 0 127 0 127 2 0.5 2"
# DATASETS["rodinia/srad/srad,large"]="16384 16384 0 127 0 127 2 0.5 2"
# DATASETS["rodinia/streamcluster/sc_omp,small"]="10 20 256 16384 65536 1000 none output.txt 12"
# DATASETS["rodinia/streamcluster/sc_omp,large"]="10 20 256 65536 65536 1000 none output.txt 12"
DATASETS["bots/alignment_for/alignment.$HCLIB_PERF_CC.for-omp-tasks,small"]="-f $BOTS_ROOT/inputs/alignment/prot.20.aa"
DATASETS["bots/alignment_for/alignment.$HCLIB_PERF_CC.for-omp-tasks,large"]="-f $BOTS_ROOT/inputs/alignment/prot.100.aa"
DATASETS["bots/alignment_single/alignment.$HCLIB_PERF_CC.single-omp-tasks,small"]="-f $BOTS_ROOT/inputs/alignment/prot.20.aa"
DATASETS["bots/alignment_single/alignment.$HCLIB_PERF_CC.single-omp-tasks,large"]="-f $BOTS_ROOT/inputs/alignment/prot.100.aa"
DATASETS["bots/fft/fft.$HCLIB_PERF_CC.omp-tasks,small"]="-n 33554432"
DATASETS["bots/fft/fft.$HCLIB_PERF_CC.omp-tasks,large"]="-n 67108864"
DATASETS["bots/fib/fib.$HCLIB_PERF_CC.omp-tasks,small"]="-n 30"
DATASETS["bots/fib/fib.$HCLIB_PERF_CC.omp-tasks,large"]="-n 34"
DATASETS["bots/floorplan/floorplan.$HCLIB_PERF_CC.omp-tasks,small"]="-f $BOTS_ROOT/inputs/floorplan/input.5"
DATASETS["bots/floorplan/floorplan.$HCLIB_PERF_CC.omp-tasks,large"]="-f $BOTS_ROOT/inputs/floorplan/input.20"
DATASETS["bots/health/health.$HCLIB_PERF_CC.omp-tasks,small"]="-f $BOTS_ROOT/inputs/health/small.input"
DATASETS["bots/health/health.$HCLIB_PERF_CC.omp-tasks,large"]="-f $BOTS_ROOT/inputs/health/medium.input"
DATASETS["bots/nqueens/nqueens.$HCLIB_PERF_CC.omp-tasks,small"]="-n 11"
DATASETS["bots/nqueens/nqueens.$HCLIB_PERF_CC.omp-tasks,large"]="-n 13"
DATASETS["bots/sort/sort.$HCLIB_PERF_CC.omp-tasks,small"]="-n 33554432"
DATASETS["bots/sort/sort.$HCLIB_PERF_CC.omp-tasks,large"]="-n 100000000"
DATASETS["bots/sparselu_for/sparselu.$HCLIB_PERF_CC.for-omp-tasks,small"]="-n 50"
DATASETS["bots/sparselu_for/sparselu.$HCLIB_PERF_CC.for-omp-tasks,large"]="-n 100"
DATASETS["bots/sparselu_single/sparselu.$HCLIB_PERF_CC.single-omp-tasks,small"]="-n 50"
DATASETS["bots/sparselu_single/sparselu.$HCLIB_PERF_CC.single-omp-tasks,large"]="-n 100"
DATASETS["bots/strassen/strassen.$HCLIB_PERF_CC.omp-tasks,small"]="-n 1024"
DATASETS["bots/strassen/strassen.$HCLIB_PERF_CC.omp-tasks,large"]="-n 4096"
# DATASETS["kastors-1.1/jacobi/jacobi-task,small"]="-c -i 100"
# DATASETS["kastors-1.1/jacobi/jacobi-task,large"]="-c -i 200"
# DATASETS["kastors-1.1/jacobi/jacobi-block-task,small"]="-c -i 100"
# DATASETS["kastors-1.1/jacobi/jacobi-block-task,large"]="-c -i 200"
# DATASETS["kastors-1.1/jacobi/jacobi-block-for,small"]="-c -i 100"
# DATASETS["kastors-1.1/jacobi/jacobi-block-for,large"]="-c -i 200"

STYLES='tied.recursive tied.flat untied.flat untied.recursive'

function setup_data_directories() {
    if [[ -z "$RODINIA_DATA_DIR" ]]; then
        echo RODINIA_DATA_DIR must be set
        exit 1
    fi

    rm -r -f $RODINIA_DATA_DIR
    mkdir -p $(dirname $RODINIA_DATA_DIR)

    OLD_DIR=$(pwd)
    cd /tmp/
    rm -f -r rodinia_3.1*
    wget http://www.cs.virginia.edu/~kw5na/lava/Rodinia/Packages/Current/rodinia_3.1.tar.bz2
    tar xf rodinia_3.1.tar.bz2
    cd $OLD_DIR

    cp -rf /tmp/rodinia_3.1/data $RODINIA_DATA_DIR

    # BFS
    echo Generating BFS datasets
    cd $RODINIA_DATA_DIR/bfs/inputGen
    cat graphgen.cpp | grep -v 'define LINEAR' | grep -v 'define UNIFORM' > tmp
    echo '#define LINEAR_CONGRUENTIAL_ENGINE linear_congruential_engine' > graphgen.cpp
    echo '#define UNIFORM_INT_DISTRIBUTION uniform_int_distribution' >> graphgen.cpp
    cat tmp >> graphgen.cpp
    make
    ./graphgen 16777216 16M
    mv graph16M.txt ../

    # Hotspot
    echo Generating Hotspot datasets
    cd $RODINIA_DATA_DIR/hotspot/inputGen
    cat hotspotex.cpp | grep -v 'include "' > tmp
    echo '#include "1024_8192.h"' > hotspotex.cpp
    cat tmp >> hotspotex.cpp
    make
    cd ../
    ./inputGen/hotspotex

    cd $OLD_DIR
}
