#!/bin/bash

time for x in $(cat benchmarks.txt); do
    pushd $x
    make clean all
    time ./run.sh
    popd
done
