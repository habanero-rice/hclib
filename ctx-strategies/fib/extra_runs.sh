#!/bin/bash

echo Using ${HCLIB_WORKERS:=8} workers

export OMP_NUM_THREADS=$HCLIB_WORKERS
export CILK_NWORKERS=$HCLIB_WORKERS
export MY_PREFIXES="omp cilk"
./run.sh

export PROJECT_RUN_ARGS="--hpx-threads=$HCLIB_WORKERS --hpx-stacksize=$((1<<18)) 35"
export MY_PREFIXES="hpx"
./run.sh
