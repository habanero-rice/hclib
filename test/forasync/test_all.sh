#!/bin/bash

set -e

for FOLDER in $(ls); do
    if [[ -d $FOLDER ]]; then
        cd $FOLDER
        make clean
        make -j

        cd ..
    fi
done


echo "========== Running arrayadd1d =========="
./arrayadd1d/arrayadd1d 1024 100

echo "========== Running arrayadd2d =========="
./arrayadd2d/arrayadd2d 1024 1024 100 50

echo "========== Running arrayadd3d =========="
./arrayadd3d/arrayadd3d 1024 1024 2048 100 50 77
