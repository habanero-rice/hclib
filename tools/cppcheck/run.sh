#!/bin/bash

set -e

cppcheck --force -I ../../inc -I ../../src/inc -I /usr/include --enable=all ../../src ../../test 1> cppcheck.log 2> cppcheck.err
