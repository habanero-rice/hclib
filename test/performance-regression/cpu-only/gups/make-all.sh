#!/bin/bash

set -e

for FOLDER in gups-ref gups-shmem gups-shmem-tasks; do
    cd $FOLDER
    make clean
    make
    cd ..
done
