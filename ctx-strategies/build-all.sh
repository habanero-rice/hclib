#!/bin/bash

set -ex

time for x in $(cat benchmarks.txt); do
    make -j -C $x clean all
done
