#!/bin/bash

[ -z "$PROJECT_NAME" ] && PROJECT_NAME=$(basename $PWD)

set -x

do_test() {
    if ! [ -f $1 ]; then
        eval make $1 $PROJECT_MAKE_ARGS
    fi
    eval timeout 60s ./$1 $HC_BIN_FLAGS $PROJECT_RUN_ARGS
    if [ "$PROJECT_VERIFY" ]; then
        eval $PROJECT_VERIFY
    fi
}


if [ -z "$HCLIB_WORKERS" ]; then
    export HCLIB_WORKERS=8
fi

mkdir -p log

for prefix in ${MY_PREFIXES:-f fh t th nb gh hclang}; do
    if [ $prefix = hclang ]; then
        # --wf -> work first
        # --hf -> help first
        export HC_BIN_FLAGS="--nproc $HCLIB_WORKERS --hf" 
    fi
    for i in {1..30}; do
        echo "===> $prefix $i"
        do_test ${prefix}_${PROJECT_NAME}
    done &>log/${prefix}_${HCLIB_WORKERS}_data.txt
done
