#!/bin/bash

DEFAULT_INPUT_DIR=./input/Cholesky
HC_EXAMPLE_INPUT_URL="https://svn.rice.edu/r/parsoft/Intel/CnC-X10/examples-input/Cholesky"

#Checking if env variable tell us where the input file is
if [[ ! -d ${DEFAULT_INPUT_DIR} ]]; then
    #No, try to fetch them locally
    svn co ${HC_EXAMPLE_INPUT_URL} ${DEFAULT_INPUT_DIR}
fi
