#!/bin/bash

set -e

make clean
make -j

export DYLD_LIBRARY_PATH=$HCLIB_HOME/modules/system/lib:$DYLD_LIBRARY_PATH

for f in `cat targets.txt`; do
    [[ -x $f && ! -d $f ]]
    echo "========== Running $f =========="
    ./$f
    echo
done
