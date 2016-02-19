#!/bin/bash

set -e

for INCLUDE_FILE in $(find ../../inc -name "*.h"); do
    FILENAME=$(basename $INCLUDE_FILE)
    indent --parameter-indentation 8 --indent-level 4 --line-length 80 $INCLUDE_FILE -o $FILENAME
done
