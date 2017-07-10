#!/bin/bash

time for x in $(cat benchmarks.txt); do
    make -j -C $x clean all
done
