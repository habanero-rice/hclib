#!/bin/bash

set -e

for f in $(find . -name "*"); do
    if [[ -x $f && ! -d $f && $(basename $f) != 'test_all.sh' ]]; then
        echo "========== Running $f =========="
        HCPP_HPT_FILE=$HCPP_HOME/hpt/hpt-max-macbook.xml $f
        echo
    fi
done
