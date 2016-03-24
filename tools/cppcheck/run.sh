#!/bin/bash

set -e

INPUT="../../src"

if [[ $# == 1 ]]; then
    INPUT=$1
fi

cppcheck --force -I ../../inc -I ../../src/inc -I /usr/include --enable=all $INPUT 1> cppcheck.log 2> cppcheck.err
