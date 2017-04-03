#!/bin/bash

set -e

if [ "$1" != "--skip-make" ]; then
    make clean
    make -j
fi

for f in `cat targets.txt`; do
    [[ -x $f && ! -d $f ]]
    echo "========== Running $f =========="
    ./$f
    echo
done
