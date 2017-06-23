#!/bin/bash

do_test() {
    make $1
    rm -f cholesky.out

    ./$1 $HC_BIN_FLAGS 500 5 ./input/m_500.in

    if ! cmp -s cholesky.out input/cholesky_out_500.txt; then
        echo "Test=Fail"
        exit 1
    fi

    echo "Test=Success"
}

export HCLIB_WORKERS=4
do_test cholesky
do_test nb_cholesky
HC_BIN_FLAGS="-nproc $HCLIB_WORKERS" do_test hclang_cholesky
