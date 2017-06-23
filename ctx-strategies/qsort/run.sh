#!/bin/bash

do_test() {
    make $1
    ./$1 $HC_BIN_FLAGS
}

export HCLIB_WORKERS=4
do_test qsort
do_test nb_qsort
HC_BIN_FLAGS="-nproc $HCLIB_WORKERS" do_test hclang_qsort
