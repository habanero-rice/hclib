#!/bin/bash

set -e

MY_OS=$(uname -s)
if [ $MY_OS = "Darwin" ]; then
    export DYLD_LIBRARY_PATH=$HCLIB_ROOT/lib:$DYLD_LIBRARY_PATH
    export DYLD_LIBRARY_PATH=$HCLIB_HOME/modules/system/lib:$DYLD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$HCLIB_ROOT/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$HCLIB_HOME/modules/system/lib:$LD_LIBRARY_PATH
fi

make clean
make -j

for f in `cat targets.txt`; do
    [[ -x $f && ! -d $f ]]
    echo "========== Running $f =========="
    ./$f
    echo
done
