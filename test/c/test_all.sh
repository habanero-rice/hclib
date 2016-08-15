#!/bin/bash

set -e

make clean
make -j

for f in `cat targets.txt`; do
    [[ -x $f && ! -d $f ]]
    echo "========== Running $f =========="
    ./$f
    echo
done
