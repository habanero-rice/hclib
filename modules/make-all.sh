#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for DIR in $(ls $SCRIPT_DIR); do
    PATH=$SCRIPT_DIR/$DIR
    if [[ -d $PATH ]]; then
        cd $PATH
        make clean
        make
        cd ..
    fi
done
