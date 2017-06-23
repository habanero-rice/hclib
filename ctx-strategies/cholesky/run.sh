#!/bin/bash

do_verify() {
    if ! cmp -s cholesky.out input/cholesky_out_500.txt; then
        echo "Test=Fail"
        exit 1
    fi
    echo OK
    rm -f cholesky.out
}

export PROJECT_RUN_ARGS="500 5 ./input/m_500.in"
export PROJECT_VERIFY=do_verify

source ../common/run.sh
