#!/bin/bash

[ -z "$PROJECT_NAME" ] && PROJECT_NAME=$(basename $PWD)

do_test() {
    set -x
    eval make $1 $PROJECT_MAKE_ARGS
    eval ./$1 $HC_BIN_FLAGS $PROJECT_RUN_ARGS
    if [ "$PROJECT_VERIFY" ]; then
        eval $PROJECT_VERIFY
    fi
}

export HCLIB_WORKERS=4
do_test $PROJECT_NAME
do_test nb_$PROJECT_NAME
# --wf -> work first
# --hf -> help first
HC_BIN_FLAGS="--nproc $HCLIB_WORKERS --hf" do_test hclang_$PROJECT_NAME
