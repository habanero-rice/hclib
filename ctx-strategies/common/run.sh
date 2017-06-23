#!/bin/bash

PROJECT_NAME=$(basename $PWD)

do_test() {
    eval make $1 $PROJECT_MAKE_ARGS
    eval ./$1 $HC_BIN_FLAGS $PROJECT_RUN_ARGS
}

export HCLIB_WORKERS=4
do_test $PROJECT_NAME
do_test nb_$PROJECT_NAME
HC_BIN_FLAGS="-nproc $HCLIB_WORKERS" do_test hclang_$PROJECT_NAME
