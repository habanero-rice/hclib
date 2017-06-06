#!/bin/bash

set -e

export DYLD_LIBRARY_PATH=$HCLIB_HOME/modules/system/lib:$DYLD_LIBRARY_PATH

make clean
make -j

for f in `cat targets.txt`; do
    [[ -x $f && ! -d $f ]]
    echo "========== Running $f =========="
    ./$f
    echo
done
