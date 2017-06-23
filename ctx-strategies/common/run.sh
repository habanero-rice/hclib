#!/bin/bash

PROJECT_NAME=$(basename $PWD)

do_test() {
    set -x
    eval make $1 $PROJECT_MAKE_ARGS
    eval ./$1 $HC_BIN_FLAGS $PROJECT_RUN_ARGS
    [ "$PROJECT_VERIFY" ] && eval $PROJECT_VERIFY
}

export HCLIB_WORKERS=4
do_test $PROJECT_NAME
do_test nb_$PROJECT_NAME
# --wf -> work first
# --hf -> help first
HC_BIN_FLAGS="--nproc $HCLIB_WORKERS --hf" do_test hclang_$PROJECT_NAME
