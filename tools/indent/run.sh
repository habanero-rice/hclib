#!/bin/bash

set -e

mkdir -p temp

OPTIONS="--parameter-indentation 8 --indent-level 4 --line-length 80 \
    --no-space-after-parentheses --braces-on-if-line --braces-on-func-def-line \
    --braces-on-struct-decl-line --dont-break-procedure-type \
    --no-space-after-function-call-names --no-tabs --dont-line-up-parentheses \
    --cuddle-else --no-space-after-casts"

for INCLUDE_FILE in $(find ../../inc -name "*.h"); do
    FILENAME=$(basename $INCLUDE_FILE)
    indent $OPTIONS  $INCLUDE_FILE -o temp/$FILENAME
    cp temp/$FILENAME $INCLUDE_FILE
    rm temp/$FILENAME
done

for FOLDER in ../../src ../../test; do
    for SRC_FILE in $(find $FOLDER -name "*.c"); do
        FILENAME=$(basename $SRC_FILE)
        indent $OPTIONS $SRC_FILE -o temp/$FILENAME
        cp temp/$FILENAME $SRC_FILE
        rm temp/$FILENAME
    done

    for SRC_FILE in $(find $FOLDER -name "*.cpp"); do
        FILENAME=$(basename $SRC_FILE)
        indent $OPTIONS $SRC_FILE -o temp/$FILENAME
        cp temp/$FILENAME $SRC_FILE
        rm temp/$FILENAME
    done
done
