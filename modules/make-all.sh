#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for DIR in $(ls $SCRIPT_DIR); do
    P=$SCRIPT_DIR/$DIR
    if [[ -d $P ]]; then
        cd $P
        make clean
        make
        cd ..
    fi
done
