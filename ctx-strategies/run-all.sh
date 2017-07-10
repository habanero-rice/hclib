#!/bin/bash

time for x in $(cat benchmarks.txt); do
    pushd $x
    time ./run.sh
    popd
done
