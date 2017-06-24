#!/bin/bash

time for x in cholesky fib needleman-wunsch nqueens qsort uts; do
    pushd $x
    make clean all
    time ./run.sh
    popd
done
