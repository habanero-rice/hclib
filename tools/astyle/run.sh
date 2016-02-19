#!/bin/bash

set -e

mkdir -p temp

OPTIONS="--style=kr --style=attach --indent=spaces=4 --max-code-length=80 \
    --align-pointer=name --break-after-logical"

for REGIX in '*.c' '*.cpp'; do
    for SRC_FILE in $(find ../../src -name "$REGIX"); do
        echo $SRC_FILE
        FILENAME=$(basename $SRC_FILE)
        astyle  $OPTIONS < $SRC_FILE > temp/$FILENAME

        cp temp/$FILENAME $SRC_FILE
        rm temp/$FILENAME
    done
done
