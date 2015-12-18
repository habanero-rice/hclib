#!/bin/bash

set -e

make clean
make -j

for f in $(find . -name "*"); do
    if [[ -x $f && ! -d $f && $(basename $f) != 'test_all.sh' ]]; then
        echo "========== Running $f =========="
        $f
        echo
    fi
done
